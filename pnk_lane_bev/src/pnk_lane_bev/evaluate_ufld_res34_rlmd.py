"""Run UFLD-v2 CULane ResNet-34 on a deterministic quarter of RLMD val."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .metrics import tolerant_lane_metrics
from .ufld_v2 import LaneRunner, UFLDDetector


LANE_IDS = set(range(3, 11)) | {16}
STOP_LINE_ID = 3
CANONICAL_WH = (768, 432)


def read_colors(csv_path: Path) -> dict[int, tuple[int, int, int]]:
    result = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) >= 5:
                result[int(row[0])] = tuple(map(int, row[2:5]))
    return result


def decode_labels(path: Path, colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"Could not read label mask: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    labels = np.zeros(rgb.shape[:2], np.uint8)
    for class_id, color in colors.items():
        if class_id:
            labels[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)] = class_id
    return labels


def rasterize(lanes, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    thickness = max(2, round(width / CANONICAL_WH[0] * 4))
    for lane in lanes:
        points = np.asarray(lane, dtype=np.float32).copy()
        if len(points):
            points[:, 0] *= width / 1600.0
            # Official UFLD-v2 CULane TensorRT demo uses a 1600x903 resize,
            # then sends rows [380:700] (height 320) to the network.
            points[:, 1] = (points[:, 1] + 380.0) * height / 903.0
            points = np.rint(points).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(mask, [points[:, None, :]], False, 255, thickness, cv2.LINE_AA)
    return mask > 0


def exact_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    union = tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "iou": tp / union if union else 1.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "precision": precision,
        "recall": recall,
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "pred_pixels": int(p.sum()),
        "gt_pixels": int(g.sum()),
    }


def aggregate(rows: list[dict], prefix: str) -> dict:
    count_keys = ("tp_pixels", "fp_pixels", "fn_pixels", "pred_pixels", "gt_pixels")
    totals = {key: sum(row[key] for row in rows) for key in count_keys}
    tp, fp, fn = totals["tp_pixels"], totals["fp_pixels"], totals["fn_pixels"]
    union = tp + fp + fn
    return {
        "images": len(rows),
        "gt_positive_images": sum(row["gt_pixels"] > 0 for row in rows),
        "prediction_empty_rate": sum(row["pred_pixels"] == 0 for row in rows) / max(len(rows), 1),
        "micro_iou": tp / union if union else 1.0,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
        "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
        **totals,
    }


def overlay(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    pred_only, gt_only, match = pred & ~gt, gt & ~pred, pred & gt
    tint = np.zeros_like(canvas)
    tint[pred_only] = (0, 0, 255)     # red: prediction only
    tint[gt_only] = (0, 255, 0)      # green: ground truth only
    tint[match] = (0, 255, 255)      # yellow: overlap
    selected = pred_only | gt_only | match
    canvas[selected] = 0.42 * canvas[selected] + 0.58 * tint[selected]
    return canvas.clip(0, 255).astype(np.uint8)


def prediction_overlay(rgb: np.ndarray, pred: np.ndarray) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    canvas[pred] = 0.45 * canvas[pred] + 0.55 * np.array([255, 255, 0], np.float32)
    return canvas.clip(0, 255).astype(np.uint8)


def run(dataset: Path, output: Path, checkpoint: Path, count: int = 100, device: str = "auto") -> dict:
    image_dir, label_dir = dataset / "images" / "val", dataset / "labels" / "val"
    paths = sorted(image_dir.glob("*.jpg"), key=lambda p: p.name.casefold())
    if not paths:
        raise FileNotFoundError(f"No validation images under {image_dir}")
    count = min(count, len(paths))
    # Evenly distribute the subset across the sorted validation sequence.
    indices = np.linspace(0, len(paths) - 1, count, dtype=np.int64)
    selected = [paths[int(i)] for i in indices]
    colors = read_colors(dataset / "rlmd.csv")
    absent = [p.name for p in selected if not (label_dir / f"{p.stem}.png").is_file()]
    if absent:
        raise FileNotFoundError(f"Missing paired masks, e.g. {absent[:3]}")

    output.mkdir(parents=True, exist_ok=True)
    comp_dir, pred_dir = output / "overlays" / "comparison", output / "overlays" / "prediction"
    comp_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    detector = UFLDDetector(checkpoint, device=device, backbone="resnet34", detector_name="ufld_v2_res34")
    if detector.device.type == "cpu":
        detector.torch.set_num_threads(min(4, detector.torch.get_num_threads()))
    parameter_count = sum(parameter.numel() for parameter in detector.model.parameters())

    lane_rows, stop_rows, rows, latencies = [], [], [], []
    started = time.perf_counter()
    batch_size = 8
    for batch_start in range(0, count, batch_size):
        batch_paths = selected[batch_start:batch_start + batch_size]
        rgbs, classes_batch, tensors = [], [], []
        for path in batch_paths:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError(f"Could not decode image {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(rgb, (1600, 903), interpolation=cv2.INTER_LINEAR)[380:700]
            tensor = crop.astype(np.float32) / 255.0
            tensor = (tensor - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
                [0.229, 0.224, 0.225], np.float32
            )
            tensors.append(tensor.transpose(2, 0, 1))
            rgbs.append(rgb)
            classes_batch.append(decode_labels(label_dir / f"{path.stem}.png", colors))
        input_tensor = detector.torch.from_numpy(np.stack(tensors)).to(detector.device)
        if detector.device.type == "cuda":
            detector.torch.cuda.synchronize()
        batch_started = time.perf_counter()
        with detector.torch.inference_mode():
            raw_batch = detector.model(input_tensor)
        if detector.device.type == "cuda":
            detector.torch.cuda.synchronize()
        batch_ms_each = (time.perf_counter() - batch_started) * 1000.0 / len(batch_paths)
        latencies.extend([batch_ms_each] * len(batch_paths))

        for offset, (path, rgb, classes) in enumerate(zip(batch_paths, rgbs, classes_batch)):
            raw = {key: value[offset:offset + 1] for key, value in raw_batch.items()}
            decoder = object.__new__(LaneRunner)
            decoder.torch = detector.torch
            records = LaneRunner._decode(decoder, raw, width=1600, height=320)
            lanes = [record["points"] for record in records]
            pred = rasterize(lanes, rgb.shape[1], rgb.shape[0])
            gt_lane = np.isin(classes, tuple(sorted(LANE_IDS)))
            gt_stop = classes == STOP_LINE_ID
            lane = exact_metrics(pred, gt_lane)
            stop = exact_metrics(pred, gt_stop)
            tolerant = tolerant_lane_metrics(pred, gt_lane, radius=5)
            lane_rows.append(lane)
            stop_rows.append(stop)
            rows.append({
                "image": path.name,
                "lane_iou": lane["iou"], "lane_f1_exact": lane["f1"],
                "lane_precision": lane["precision"], "lane_recall": lane["recall"],
                "lane_f1_tolerance_5px": tolerant["lane_f1"],
                "stop_line_iou_subset": stop["iou"], "stop_line_f1_subset": stop["f1"],
                "prediction_empty": int(not pred.any()),
                "gt_lane_pixels": lane["gt_pixels"], "pred_lane_pixels": lane["pred_pixels"],
                "gt_stop_pixels": stop["gt_pixels"], "runtime_ms": batch_ms_each,
                "lane_count": len(records),
            })
            cv2.imwrite(str(comp_dir / f"{path.stem}.jpg"), overlay(rgb, pred, gt_lane), [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(pred_dir / f"{path.stem}.jpg"), prediction_overlay(rgb, pred), [cv2.IMWRITE_JPEG_QUALITY, 92])
        processed = batch_start + len(batch_paths)
        print(f"[UFLD-R34-RLMD] {processed}/{count} last={batch_paths[-1].name}", flush=True)

    wall_time = time.perf_counter() - started
    lane_summary = aggregate(lane_rows, "lane")
    stop_summary = aggregate(stop_rows, "stop")
    lane_summary["mean_image_f1_tolerance_5px"] = float(np.mean([row["lane_f1_tolerance_5px"] for row in rows]))
    result = {
        "task": "UFLD-v2 ResNet-34 only on RLMD validation quarter",
        "dataset": str(dataset.resolve()), "split": "val",
        "sampling": {"policy": "evenly spaced over numeric-sorted validation filenames", "images_total": len(paths), "images_run": len(selected), "fraction": len(selected) / len(paths), "names": [p.name for p in selected]},
        "model": {"architecture": "UFLD-v2", "backbone": "ResNet-34", "training_domain": "CULane", "checkpoint": str(checkpoint.resolve()), "parameters": parameter_count, "device": str(detector.device), "drivable_model": None},
        "preprocessing_note": "Uses the official UFLD-v2 CULane deployment crop: resize to 1600x903, crop rows 380:700 (1600x320), then map predictions back to original image space.",
        "ground_truth": {"source": str((dataset / "rlmd.csv").resolve()), "lane_ids": sorted(LANE_IDS), "stop_line_id": STOP_LINE_ID, "drivable_ground_truth": None},
        "metrics": {"lane_marking": lane_summary, "generic_lane_prediction_vs_stop_line_subset": stop_summary,
                    "empty_prediction_rate": sum(row["prediction_empty"] for row in rows) / len(rows),
                    "mean_lanes_per_image": float(np.mean([row["lane_count"] for row in rows]))},
        "runtime": {"mean_batch_amortized_model_ms_per_image": float(np.mean(latencies)), "p95_batch_amortized_ms_per_image": float(np.percentile(latencies, 95)), "batch_size": batch_size, "wall_time_s": wall_time},
        "limitations": ["No Mask2Former or drivable-area model was loaded or run.", "The stop-line subset metric is exploratory: UFLD predicts generic lanes and has no stop-line class.", "A quarter-split result is a subset estimate, not a full validation benchmark."],
        "per_image": rows,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "sampling.txt").write_text("\n".join(p.name for p in selected) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# UFLD-v2 ResNet-34 on RLMD validation (25%)\n\n"
        "Standalone lane-only inference. No Mask2Former/drivable model was loaded.\n\n"
        "- `metrics.json`: aggregate and per-image results.\n"
        "- `per_image.csv`: image-level metrics.\n"
        "- `sampling.txt`: exact 100-image subset.\n"
        "- `overlays/comparison/`: red = prediction only, green = GT only, yellow = overlap.\n"
        "- `overlays/prediction/`: cyan = UFLD lane prediction on RGB.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("pnk_lane_bev/models/culane_res34.pth"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run(args.dataset, args.output, args.checkpoint, args.count, args.device)
    print(json.dumps({"metrics": result["metrics"], "runtime": result["runtime"], "model": result["model"]}, indent=2))


if __name__ == "__main__":
    main()
