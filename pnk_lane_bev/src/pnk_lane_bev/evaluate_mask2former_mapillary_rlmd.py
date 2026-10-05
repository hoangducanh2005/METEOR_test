"""Evaluate the Mapillary Mask2Former Task 2 branch on RLMD validation masks."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pnk_lane_bev.mask2former_unified import (
    MAPILLARY_ID2LABEL,
    Mask2FormerDetector,
    postprocess_lane,
)
from pnk_lane_bev.unified_config import load_unified_config, resolve_class_ids


# RLMD taxonomy: road-line paint categories (including stop line) and channelizing line.
GT_LANE_LINE_CLASS_IDS = set(range(3, 11)) | {16}
GT_CROSSWALK_CLASS_IDS = {2}
GT_STOPLINE_CLASS_IDS = {3}


def _class_color_map(csv_path: Path) -> dict[int, tuple[int, int, int]]:
    colors: dict[int, tuple[int, int, int]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) >= 5:
                colors[int(row[0])] = (int(row[2]), int(row[3]), int(row[4]))
    return colors


def _decode_label(label_path: Path, color_map: dict[int, tuple[int, int, int]]) -> np.ndarray:
    bgr = cv2.imread(str(label_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"Could not decode label {label_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    class_map = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for class_id, color in color_map.items():
        if class_id == 0:
            continue
        class_map[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)] = class_id
    return class_map


def _counts(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, int]:
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "pred_pixels": int(pred.sum()), "gt_pixels": int(gt.sum())}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    union = tp + fp + fn
    gt_positive = [row for row in rows if row["gt_pixels"] > 0]
    return {
        "images": len(rows),
        "gt_positive_images": len(gt_positive),
        "gt_empty_rate": sum(row["gt_pixels"] == 0 for row in rows) / max(1, len(rows)),
        "prediction_empty_rate": sum(row["pred_pixels"] == 0 for row in rows) / max(1, len(rows)),
        "micro_iou": tp / union if union else 1.0,
        "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
        "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "mean_iou_gt_positive_images": float(np.mean([row["iou"] for row in gt_positive])) if gt_positive else None,
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "predicted_pixels": sum(row["pred_pixels"] for row in rows),
        "gt_pixels": sum(row["gt_pixels"] for row in rows),
    }


def _comparison_overlay(rgb: np.ndarray, prediction: np.ndarray, gt: np.ndarray) -> np.ndarray:
    output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    pred = prediction.astype(bool)
    truth = gt.astype(bool)
    pred_only = pred & ~truth
    gt_only = truth & ~pred
    overlap = pred & truth
    tint = np.zeros_like(output)
    tint[pred_only] = (0, 0, 255)       # red = prediction only
    tint[gt_only] = (0, 255, 0)         # green = GT only
    tint[overlap] = (0, 255, 255)       # yellow = overlap
    selected = pred_only | gt_only | overlap
    output[selected] = 0.45 * output[selected] + 0.55 * tint[selected]
    return output.clip(0, 255).astype(np.uint8)


def _drivable_overlay(rgb: np.ndarray, drive: np.ndarray, lane: np.ndarray) -> np.ndarray:
    output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    drive_b = drive.astype(bool)
    lane_b = lane.astype(bool)
    output[drive_b] = 0.72 * output[drive_b] + 0.28 * np.array([0, 190, 0])
    output[lane_b] = 0.45 * output[lane_b] + 0.55 * np.array([0, 255, 255])
    return output.clip(0, 255).astype(np.uint8)


def run(dataset: Path, output: Path, config_path: Path, split: str = "val",
        device: str = "auto", limit: int | None = None) -> dict[str, Any]:
    image_dir = dataset / "images" / split
    label_dir = dataset / "labels" / split
    csv_path = dataset / "rlmd.csv"
    if not image_dir.is_dir() or not label_dir.is_dir() or not csv_path.is_file():
        raise FileNotFoundError(f"Expected images/{split}, labels/{split}, and rlmd.csv under {dataset}")
    image_paths = sorted(image_dir.glob("*.jpg"), key=lambda path: path.name.casefold())
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise ValueError(f"No JPG images found in {image_dir}")
    absent = [path.name for path in image_paths if not (label_dir / f"{path.stem}.png").is_file()]
    if absent:
        raise ValueError(f"{len(absent)} images lack matching labels: {absent[:5]}")

    config = load_unified_config(config_path)
    model_name = config["model"]["name"]
    cache = Path(config["model"]["local_cache_dir"])
    detector = Mask2FormerDetector(model_name, device=device, cache_dir=cache)
    id2label = detector.id2label or MAPILLARY_ID2LABEL
    drivable_ids = set(resolve_class_ids(
        id2label, config["taxonomy"]["drivable_classes"], "drivable_classes"
    ).keys())
    lane_ids = set(resolve_class_ids(
        id2label, config["taxonomy"]["lane_classes"], "lane_classes"
    ).keys())
    # The dataset has one forward-facing camera but no camera identifier. Use
    # the configured FRONT_WIDE threshold as the closest canonical profile.
    lane_threshold = float(config["cameras"]["CAM_FRONT_WIDE"]["lane_confidence_threshold"])
    lane_options = config["postprocess"]["lane"]
    colors = _class_color_map(csv_path)

    output.mkdir(parents=True, exist_ok=True)
    semantic_dir = output / "masks" / "semantic"
    lane_dir = output / "masks" / "lane_marking"
    drive_dir = output / "masks" / "drivable"
    overlay_dir = output / "overlays"
    for path in (semantic_dir, lane_dir, drive_dir, overlay_dir):
        path.mkdir(parents=True, exist_ok=True)

    lane_rows: list[dict[str, Any]] = []
    stop_rows: list[dict[str, Any]] = []
    crosswalk_rows: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    infer_ms: list[float] = []
    total_lane_pixels = total_drive_pixels = 0

    # Mapillary id2label may be index -> label; keep model-provided IDs.
    for index, image_path in enumerate(image_paths, start=1):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"Could not decode image {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt_classes = _decode_label(label_dir / f"{image_path.stem}.png", colors)
        if gt_classes.shape != rgb.shape[:2]:
            raise ValueError(f"Image/label shape mismatch at {image_path.name}")

        prediction = detector.predict(rgb)
        infer_ms.append(float(prediction.latency_ms))
        lane_raw = (
            np.isin(prediction.semantic, list(lane_ids))
            & (prediction.semantic_confidence >= lane_threshold)
        ).astype(np.uint8)
        lane_mask = postprocess_lane(lane_raw, lane_options)
        drive_mask = np.isin(prediction.semantic, list(drivable_ids)).astype(np.uint8)
        total_lane_pixels += int(lane_mask.sum())
        total_drive_pixels += int(drive_mask.sum())

        gt_lane = np.isin(gt_classes, list(GT_LANE_LINE_CLASS_IDS))
        gt_stop = np.isin(gt_classes, list(GT_STOPLINE_CLASS_IDS))
        gt_crosswalk = np.isin(gt_classes, list(GT_CROSSWALK_CLASS_IDS))
        pred_lane = lane_mask.astype(bool)
        pred_stop = pred_lane  # Mapillary has general lane-marking, not a stop-line class.
        pred_crosswalk = (
            np.isin(prediction.semantic, [8, 23])
            & (prediction.semantic_confidence >= lane_threshold)
        )
        pred_crosswalk = postprocess_lane(pred_crosswalk.astype(np.uint8), lane_options).astype(bool)

        lane_counts = _counts(pred_lane, gt_lane)
        stop_counts = _counts(pred_stop, gt_stop)
        crosswalk_counts = _counts(pred_crosswalk, gt_crosswalk)
        for target, counts in ((lane_rows, lane_counts), (stop_rows, stop_counts), (crosswalk_rows, crosswalk_counts)):
            union = counts["tp"] + counts["fp"] + counts["fn"]
            target.append({
                "iou": counts["tp"] / union if union else 1.0,
                **counts,
            })
        per_image.append({
            "image": image_path.name,
            "lane_marking": {"iou": lane_rows[-1]["iou"], **lane_counts},
            "stop_line_subset": {"iou": stop_rows[-1]["iou"], **stop_counts},
            "crosswalk": {"iou": crosswalk_rows[-1]["iou"], **crosswalk_counts},
            "drivable_prediction_pixels": int(drive_mask.sum()),
            "lane_prediction_pixels": int(lane_mask.sum()),
            "mean_model_latency_ms": float(prediction.latency_ms),
        })

        cv2.imwrite(str(semantic_dir / f"{image_path.stem}.png"), prediction.semantic.astype(np.uint16))
        cv2.imwrite(str(lane_dir / f"{image_path.stem}.png"), lane_mask * 255)
        cv2.imwrite(str(drive_dir / f"{image_path.stem}.png"), drive_mask * 255)
        cv2.imwrite(
            str(overlay_dir / f"{image_path.stem}.jpg"),
            _comparison_overlay(rgb, lane_mask, gt_lane),
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )
        if index <= 20 or index % 20 == 0:
            cv2.imwrite(
                str(output / "overlays" / f"{image_path.stem}_prediction.jpg"),
                _drivable_overlay(rgb, drive_mask, lane_mask),
                [cv2.IMWRITE_JPEG_QUALITY, 88],
            )
        if index % 20 == 0 or index == len(image_paths):
            print(
                f"[{index}/{len(image_paths)}] {image_path.name} "
                f"lane_iou={lane_rows[-1]['iou']:.4f} "
                f"drive_coverage={float(drive_mask.mean()):.3f}",
                flush=True,
            )

    result = {
        "schema_version": 1,
        "task": "task_2_mask2former_mapillary_rlmd_evaluation",
        "dataset_root": str(dataset.resolve()),
        "split": split,
        "image_count": len(image_paths),
        "resolution": "1920x1080",
        "architecture": {
            "model": model_name,
            "taxonomy": "Mapillary Vistas 65-class panoptic",
            "drivable_model_classes": config["taxonomy"]["drivable_classes"],
            "lane_model_classes": config["taxonomy"]["lane_classes"],
            "lane_confidence_threshold": lane_threshold,
            "lane_postprocess": lane_options,
            "device": str(detector.device),
        },
        "ground_truth_mapping": {
            "source": str(csv_path),
            "lane_line_class_ids": sorted(GT_LANE_LINE_CLASS_IDS),
            "lane_line_classes": ["stop line", "solid single white", "solid single yellow", "solid single red", "solid double white", "solid double yellow", "dashed single white", "dashed single yellow", "channelizing line"],
            "stop_line_class_ids": sorted(GT_STOPLINE_CLASS_IDS),
            "crosswalk_class_ids": sorted(GT_CROSSWALK_CLASS_IDS),
            "drivable_area_gt": None,
            "drivable_area_note": "RLMD has no road/drivable-area class; background is unlabeled, so drivable IoU is not valid.",
        },
        "runtime": {
            "wall_time_s": time.perf_counter() - run_started,
            "mean_model_latency_ms": float(np.mean(infer_ms)),
            "total_model_latency_s": float(np.sum(infer_ms) / 1000.0),
        },
        "metrics_definition": {
            "lane_marking": "Mapillary Lane Marking-General/Crosswalk prediction vs RLMD IDs 3-10 and 16; pixel micro metrics.",
            "stop_line": "Mapillary lane-marking prediction vs RLMD stop-line ID 3; exploratory because Mapillary has no stop-line class.",
            "crosswalk": "Mapillary Crosswalk-Plain/Lane-Marking-Crosswalk prediction vs RLMD ID 2.",
            "empty_rate": "fraction of images with an empty predicted class mask.",
            "per_image_mean_iou": "averaged only over images with non-empty GT mask.",
        },
        "metrics": {
            "lane_marking": _aggregate(lane_rows),
            "stop_line_subset": _aggregate(stop_rows),
            "crosswalk": _aggregate(crosswalk_rows),
            "drivable_area": None,
            "predicted_drivable_coverage_mean": total_drive_pixels / (len(image_paths) * 1920 * 1080),
            "predicted_lane_coverage_mean": total_lane_pixels / (len(image_paths) * 1920 * 1080),
        },
        "per_image": per_image,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("pnk_lane_bev/configs/segmentation/mapillary_swin_large.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = run(args.dataset, args.output, args.config, args.split, args.device, args.limit)
    print(json.dumps({"metrics": result["metrics"], "runtime": result["runtime"]}, indent=2))


if __name__ == "__main__":
    main()
