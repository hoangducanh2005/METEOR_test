"""Evaluate Task 3's Grounding DINO + SAM 2 stop-line branch on RLMD masks.

RLMD contains independent validation images rather than a calibrated camera
sequence, so SAM 2 image segmentation is run per image and no BEV metrics are
computed. The image model is loaded from the same SAM 2.1 Hiera Small checkpoint
used by the Task 3 video branch.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)


PROMPT = "stop line."
GROUNDING_DINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM2_ID = "facebook/sam2.1-hiera-small"
BOX_THRESHOLD = 0.22
TEXT_THRESHOLD = 0.18


def _read_label_color(csv_path: Path, class_name: str) -> tuple[int, int, int]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 5 and row[1].strip().casefold() == class_name.casefold():
                return int(row[2]), int(row[3]), int(row[4])
    raise ValueError(f"Class {class_name!r} not found in {csv_path}")


def _postprocess_stop(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    if not binary.any():
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros_like(binary)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= 20:
            keep[labels == component] = 1
    return keep


def _overlay(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    pred_only = pred.astype(bool) & ~gt.astype(bool)
    gt_only = gt.astype(bool) & ~pred.astype(bool)
    overlap = pred.astype(bool) & gt.astype(bool)
    colors = np.zeros_like(image)
    colors[pred_only] = (0, 0, 255)       # red: prediction only
    colors[gt_only] = (0, 255, 0)         # green: GT only
    colors[overlap] = (0, 255, 255)      # yellow: overlap
    active = pred_only | gt_only | overlap
    image[active] = 0.45 * image[active] + 0.55 * colors[active]
    return image.clip(0, 255).astype(np.uint8)


def _metric_counts(pred: np.ndarray, gt: np.ndarray) -> dict[str, int]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    return {
        "tp": int(np.logical_and(pred_b, gt_b).sum()),
        "fp": int(np.logical_and(pred_b, ~gt_b).sum()),
        "fn": int(np.logical_and(~pred_b, gt_b).sum()),
        "pred_pixels": int(pred_b.sum()),
        "gt_pixels": int(gt_b.sum()),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    union = tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "images": len(rows),
        "ground_truth_positive_images": sum(row["gt_pixels"] > 0 for row in rows),
        "ground_truth_empty_rate": sum(row["gt_pixels"] == 0 for row in rows) / max(1, len(rows)),
        "prediction_empty_rate": sum(row["pred_pixels"] == 0 for row in rows) / max(1, len(rows)),
        "miss_rate_on_gt_positive_images": sum(
            row["pred_pixels"] == 0 and row["gt_pixels"] > 0 for row in rows
        ) / max(1, sum(row["gt_pixels"] > 0 for row in rows)),
        "micro_iou": tp / union if union else 1.0,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "mean_iou_gt_positive_images": float(np.mean([
            row["iou"] for row in rows if row["gt_pixels"] > 0
        ])) if any(row["gt_pixels"] > 0 for row in rows) else None,
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "predicted_stopline_pixels": sum(row["pred_pixels"] for row in rows),
        "ground_truth_stopline_pixels": sum(row["gt_pixels"] for row in rows),
    }


def run(dataset: Path, output: Path, cache: Path, device: str = "auto",
        split: str = "val", limit: int | None = None) -> dict[str, Any]:
    image_dir = dataset / "images" / split
    label_dir = dataset / "labels" / split
    label_csv = dataset / "rlmd.csv"
    if not image_dir.is_dir() or not label_dir.is_dir() or not label_csv.is_file():
        raise FileNotFoundError(f"Expected images/{split}, labels/{split}, and rlmd.csv under {dataset}")
    image_paths = sorted(image_dir.glob("*.jpg"), key=lambda path: path.name.casefold())
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise ValueError(f"No JPG images found in {image_dir}")
    missing_labels = [path.name for path in image_paths if not (label_dir / f"{path.stem}.png").is_file()]
    if missing_labels:
        raise ValueError(f"{len(missing_labels)} images lack matching labels, e.g. {missing_labels[:5]}")

    stopline_rgb = _read_label_color(label_csv, "stop line")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    dtype = torch.float16 if torch_device.type == "cuda" else torch.float32
    output.mkdir(parents=True, exist_ok=True)
    pred_dir = output / "masks" / "pred_stopline"
    overlay_dir = output / "overlays"
    pred_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    gd_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_ID, cache_dir=cache, local_files_only=True)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_DINO_ID, cache_dir=cache, dtype=dtype, local_files_only=True
    ).to(torch_device).eval()
    sam_processor = Sam2Processor.from_pretrained(SAM2_ID, cache_dir=cache, local_files_only=True)
    sam_model = Sam2Model.from_pretrained(
        SAM2_ID, cache_dir=cache, dtype=dtype, local_files_only=True
    ).to(torch_device).eval()

    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for index, image_path in enumerate(image_paths):
        rgb = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if rgb is None:
            raise OSError(f"Could not decode {image_path}")
        height, width = rgb.shape[:2]
        label_path = label_dir / f"{image_path.stem}.png"
        label_rgb = cv2.cvtColor(cv2.imread(str(label_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if label_rgb is None or label_rgb.shape[:2] != (height, width):
            raise ValueError(f"Image/label shape mismatch for {image_path.name}")
        gt = np.all(label_rgb == np.asarray(stopline_rgb, dtype=np.uint8), axis=2).astype(np.uint8)

        start = time.perf_counter()
        pil_image = Image.fromarray(rgb)
        gd_inputs = gd_processor(images=pil_image, text=PROMPT, return_tensors="pt").to(torch_device)
        gd_inputs["pixel_values"] = gd_inputs["pixel_values"].to(dtype=dtype)
        with torch.inference_mode():
            gd_outputs = gd_model(**gd_inputs)
        grounded = gd_processor.post_process_grounded_object_detection(
            gd_outputs,
            gd_inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(height, width)],
        )[0]
        boxes = grounded["boxes"].detach().float().cpu().numpy().tolist()
        scores = grounded["scores"].detach().float().cpu().numpy().tolist()

        pred = np.zeros((height, width), dtype=np.uint8)
        if boxes:
            union_box = [
                min(box[0] for box in boxes), min(box[1] for box in boxes),
                max(box[2] for box in boxes), max(box[3] for box in boxes),
            ]
            sam_inputs = sam_processor(
                images=pil_image, input_boxes=[[union_box]], return_tensors="pt"
            ).to(torch_device)
            sam_inputs["pixel_values"] = sam_inputs["pixel_values"].to(dtype=dtype)
            with torch.inference_mode():
                sam_output = sam_model(**sam_inputs, multimask_output=False)
            logits = sam_processor.post_process_masks(
                [sam_output.pred_masks[0]], original_sizes=[[height, width]], binarize=False
            )[0]
            raw_mask = logits[0, 0].detach().float().cpu().numpy() >= 0.0
            pred = _postprocess_stop(raw_mask)

        counts = _metric_counts(pred, gt)
        union = counts["tp"] + counts["fp"] + counts["fn"]
        row = {
            "image": image_path.name,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "width": width,
            "height": height,
            "grounding_box_count": len(boxes),
            "grounding_scores": scores,
            "prediction_empty": counts["pred_pixels"] == 0,
            "ground_truth_empty": counts["gt_pixels"] == 0,
            "iou": counts["tp"] / union if union else 1.0,
            "latency_ms": (time.perf_counter() - start) * 1000.0,
            **counts,
        }
        rows.append(row)
        cv2.imwrite(str(pred_dir / f"{image_path.stem}.png"), pred * 255)
        cv2.imwrite(str(overlay_dir / f"{image_path.stem}.jpg"), _overlay(rgb, pred, gt), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if (index + 1) % 20 == 0 or index + 1 == len(image_paths):
            print(f"[{index + 1}/{len(image_paths)}] {image_path.name} IoU={row['iou']:.4f} pred_empty={row['prediction_empty']}", flush=True)

    result = {
        "schema_version": 1,
        "task": "task_4a_rlmd_grounded_sam2_stopline_evaluation",
        "dataset_root": str(dataset.resolve()),
        "split": split,
        "image_count": len(image_paths),
        "label_mapping": {"class": "stop line", "rgb": list(stopline_rgb), "id": 3},
        "architecture": {
            "grounding_dino": GROUNDING_DINO_ID,
            "sam2": SAM2_ID,
            "prompt": PROMPT,
            "box_threshold": BOX_THRESHOLD,
            "text_threshold": TEXT_THRESHOLD,
            "sam_inference_mode": "independent image segmentation using SAM 2.1 checkpoint; temporal propagation disabled",
            "mask_threshold": 0.0,
            "postprocessing": "5x3 morphological close; remove components under 20 pixels",
        },
        "device": str(torch_device),
        "runtime": {
            "wall_time_s": time.perf_counter() - started_all,
            "mean_latency_ms_per_image": float(np.mean([row["latency_ms"] for row in rows])),
        },
        "metrics_definition": {
            "micro_iou": "sum(TP pixels) / sum(TP + FP + FN) over all validation images",
            "mean_iou_gt_positive_images": "per-image IoU averaged only where GT has at least one stop-line pixel",
            "prediction_empty_rate": "fraction of images where predicted mask has zero pixels",
            "miss_rate_on_gt_positive_images": "fraction of GT-positive images with an empty prediction",
            "pixel_color_order": "PNG labels decoded as RGB and matched exactly to rlmd.csv class color",
        },
        "metrics": _aggregate(rows),
        "per_image": rows,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("out/grounded_sam2/hf_cache"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = run(args.dataset, args.output, args.cache, args.device, args.split, args.limit)
    print(json.dumps({"metrics": result["metrics"], "runtime": result["runtime"]}, indent=2))


if __name__ == "__main__":
    main()
