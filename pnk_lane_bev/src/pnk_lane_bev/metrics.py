"""Proxy-reference and self-consistency evaluation for Task 2."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .constants import (
    BEV_DRIVABLE_IDS,
    BEV_LANE_ID,
    BEV_RESOLUTION,
    CAMERAS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SEG2D_LANE_ID,
    SEG2D_ROAD_ID,
)
from .io import atomic_json, load_task1_manifest, read_json
from .project import bev_stem, lane_raster


def _warp_previous_to_current(mask: np.ndarray, previous_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
    """Warp a previous ego-grid raster into the current ego frame using NAV poses."""
    rows, cols = np.indices(mask.shape, dtype=np.float32)
    current_x = 80.0 - (rows + 0.5) * BEV_RESOLUTION
    current_y = 50.0 - (cols + 0.5) * BEV_RESOLUTION
    current_yaw, previous_yaw = float(current_pose[2]), float(previous_pose[2])
    cos_c, sin_c = np.cos(current_yaw), np.sin(current_yaw)
    world_x = cos_c * current_x - sin_c * current_y + float(current_pose[0])
    world_y = sin_c * current_x + cos_c * current_y + float(current_pose[1])
    dx, dy = world_x - float(previous_pose[0]), world_y - float(previous_pose[1])
    cos_p, sin_p = np.cos(previous_yaw), np.sin(previous_yaw)
    previous_x = cos_p * dx + sin_p * dy
    previous_y = -sin_p * dx + cos_p * dy
    map_row = (80.0 - previous_x) / BEV_RESOLUTION - 0.5
    map_col = (50.0 - previous_y) / BEV_RESOLUTION - 0.5
    return cv2.remap(
        mask.astype(np.uint8), map_col.astype(np.float32), map_row.astype(np.float32),
        cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)


def _safe(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def binary_metrics(prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray | None = None) -> dict:
    prediction, reference = prediction.astype(bool), reference.astype(bool)
    if valid is None:
        valid = np.ones_like(prediction, dtype=bool)
    else:
        valid = valid.astype(bool)
    tp = int((prediction & reference & valid).sum())
    fp = int((prediction & ~reference & valid).sum())
    fn = int((~prediction & reference & valid).sum())
    return {
        "iou": _safe(tp / max(tp + fp + fn, 1)),
        "dice": _safe(2 * tp / max(2 * tp + fp + fn, 1)),
        "precision": _safe(tp / max(tp + fp, 1)),
        "recall": _safe(tp / max(tp + fn, 1)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sample_count": int(valid.sum()),
    }


def tolerant_lane_metrics(prediction: np.ndarray, reference: np.ndarray, radius: int) -> dict:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    pred, ref = prediction.astype(np.uint8), reference.astype(np.uint8)
    pred_dilated, ref_dilated = cv2.dilate(pred, kernel), cv2.dilate(ref, kernel)
    matched_pred = int(((pred > 0) & (ref_dilated > 0)).sum())
    matched_ref = int(((ref > 0) & (pred_dilated > 0)).sum())
    pred_count, ref_count = int(pred.sum()), int(ref.sum())
    precision = matched_pred / max(pred_count, 1)
    recall = matched_ref / max(ref_count, 1)
    return {
        "lane_precision": _safe(precision),
        "lane_recall": _safe(recall),
        "lane_f1": _safe(2 * precision * recall / max(precision + recall, 1e-9)),
        "predicted_lane_pixels": pred_count,
        "reference_lane_pixels": ref_count,
    }


def chamfer_m(prediction: np.ndarray, reference: np.ndarray) -> float | None:
    pred, ref = prediction.astype(bool), reference.astype(bool)
    if not pred.any() or not ref.any():
        return None
    to_ref = cv2.distanceTransform((~ref).astype(np.uint8), cv2.DIST_L2, 3)
    to_pred = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, 3)
    return float(0.5 * (to_ref[pred].mean() + to_pred[ref].mean()) * BEV_RESOLUTION)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=np.float64)
    if not len(values):
        return {"mean": None, "median": None, "p95": None, "count": 0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "count": int(len(values)),
    }


def _chart(path: Path, title: str, labels: list[str], values: list[float], color=(65, 190, 255)) -> None:
    canvas = np.full((480, 900, 3), 24, np.uint8)
    cv2.putText(canvas, title, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
    maximum = max(max(values, default=1), 1e-6)
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 85 + index * 60
        cv2.putText(canvas, label.replace("CAM_", ""), (30, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (215, y), (815, y + 30), (50, 50, 50), -1)
        cv2.rectangle(canvas, (215, y), (215 + int(600 * value / maximum), y + 30), color, -1)
        cv2.putText(canvas, f"{value:.3f}", (825, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def evaluate(scene_dir: Path, output_dir: Path, limit: int | None = None) -> dict:
    manifest = load_task1_manifest(scene_dir)
    frames = manifest["frames"][:limit] if limit is not None else manifest["frames"]
    camera_order = list(manifest["cams"])
    image_rows, bev_rows = [], []
    previous_drive, previous_lane = None, None
    previous_pose = None
    with np.load(scene_dir / "ego_motion.npz") as motion:
        poses = motion["pose"].astype(np.float64)
    for frame_index, frame in enumerate(frames, 1):
        with np.load(scene_dir / frame["seg2d21"]) as data:
            proxy_semantic = data["seg"]
        for camera in CAMERAS:
            camera_index = camera_order.index(camera)
            proxy = cv2.resize(proxy_semantic[camera_index], (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
            with np.load(output_dir / "cache" / "mask2former_small" / camera / f"{int(frame['frame']):04d}.npz") as data:
                drive_pred = data["mask"].astype(bool)
                drive_prob = data["probability"].astype(np.float32)
            lane_record = read_json(output_dir / "cache" / "ufld_v2" / camera / f"{int(frame['frame']):04d}.json")
            lane_pred = lane_raster(lane_record) > 0.20
            drive_ref, lane_ref = proxy == SEG2D_ROAD_ID, proxy == SEG2D_LANE_ID
            metrics = binary_metrics(drive_pred, drive_ref)
            metrics.update(tolerant_lane_metrics(lane_pred, lane_ref, radius=5))
            metrics.update(
                {
                    "record_type": "image",
                    "reference_type": "PROXY_TASK1_SEG2D21",
                    "reference_id": "Task1 seg2d21 road=11 lane=13",
                    "frame": int(frame["frame"]),
                    "timestamp": int(frame["timestamp"]),
                    "camera": camera,
                    "drivable_fraction": float(drive_pred.mean()),
                    "mean_road_probability": float(drive_prob.mean()),
                    "brier": float(np.mean((drive_prob - drive_ref.astype(np.float32)) ** 2)),
                    "lane_count": len(lane_record.get("lanes", [])),
                    "mask2former_latency_ms": float(read_json(output_dir / "cache" / "mask2former_small" / camera / f"{int(frame['frame']):04d}.json")["latency_ms"]),
                    "ufld_latency_ms": float(lane_record["latency_ms"]),
                }
            )
            image_rows.append(metrics)

        stem = bev_stem(frame)
        with np.load(output_dir / "bev" / f"{stem}.npz") as data:
            drive = data["drivable_mask"].astype(bool)
            lane = data["lane_mask"].astype(bool)
            valid = data["coverage"] > 0
            per_drive = data["per_camera_drivable"].astype(np.float32)
            per_valid = data["per_camera_valid"].astype(bool)
        proxy_bev = cv2.imread(str(scene_dir / frame["gt_vec"]), cv2.IMREAD_GRAYSCALE)
        if proxy_bev is None:
            raise FileNotFoundError(scene_dir / frame["gt_vec"])
        drive_ref = np.isin(proxy_bev, BEV_DRIVABLE_IDS)
        lane_ref = proxy_bev == BEV_LANE_ID
        bev_metric = {f"drive_{key}": value for key, value in binary_metrics(drive, drive_ref, valid).items()}
        bev_metric.update(tolerant_lane_metrics(lane & valid, lane_ref & valid, radius=2))
        bev_metric["lane_chamfer_m"] = chamfer_m(lane & valid, lane_ref & valid)
        overlaps = []
        for first in range(len(CAMERAS)):
            for second in range(first + 1, len(CAMERAS)):
                overlap = per_valid[first] & per_valid[second]
                if int(overlap.sum()) > 100:
                    overlaps.append(binary_metrics(per_drive[first] >= 0.42, per_drive[second] >= 0.42, overlap)["iou"])
        current_pose = poses[int(frame["frame"])]
        warped_previous_drive = (
            _warp_previous_to_current(previous_drive, previous_pose, current_pose)
            if previous_drive is not None else None
        )
        warped_previous_lane = (
            _warp_previous_to_current(previous_lane, previous_pose, current_pose)
            if previous_lane is not None else None
        )
        bev_metric.update(
            {
                "record_type": "bev",
                "reference_type": "PROXY_TASK1_GT_VEC",
                "reference_id": "Task1 gt_vec drivable=1,3,4,5,7,8 lane=4",
                "frame": int(frame["frame"]),
                "timestamp": int(frame["timestamp"]),
                "camera": "FUSED_BEV",
                "valid_coverage": float(valid.mean()),
                "cross_camera_drive_iou_self": float(np.mean(overlaps)) if overlaps else None,
                "temporal_drive_iou_self": binary_metrics(drive, warped_previous_drive, valid)["iou"] if warped_previous_drive is not None else None,
                "temporal_lane_chamfer_m_self": chamfer_m(lane & valid, warped_previous_lane & valid) if warped_previous_lane is not None else None,
                "temporal_warp": "NAV ego_motion pose t-1 to t",
            }
        )
        previous_drive, previous_lane, previous_pose = drive, lane, current_pose
        bev_rows.append(bev_metric)
        print(f"[evaluate] {frame_index}/{len(frames)} frame={frame['frame']}", flush=True)

    all_rows = image_rows + bev_rows
    _write_csv(output_dir / "metrics" / "frame_metrics.csv", all_rows)
    camera_rows = []
    for camera in CAMERAS:
        subset = [row for row in image_rows if row["camera"] == camera]
        row = {"camera": camera, "reference_type": "PROXY_TASK1_SEG2D21", "sample_count": len(subset)}
        for key in ("iou", "dice", "precision", "recall", "lane_f1", "brier", "mask2former_latency_ms", "ufld_latency_ms", "lane_count"):
            stats = _aggregate(subset, key)
            for stat, value in stats.items():
                row[f"{key}_{stat}"] = value
        camera_rows.append(row)
    _write_csv(output_dir / "metrics" / "camera_summary.csv", camera_rows)

    geometry = read_json(output_dir / "metrics" / "geometry.json")
    mask_runtime = _aggregate(image_rows, "mask2former_latency_ms")
    lane_runtime = _aggregate(image_rows, "ufld_latency_ms")
    projection_summary = read_json(output_dir / "projection_summary.json")
    scorecard = [
        {"architecture": "Mask2Former Swin-Small", "reference_type": "PROXY_TASK1_SEG2D21", "primary_metric": "road_iou", "value": _aggregate(image_rows, "iou")["mean"], "samples": len(image_rows), "runtime_median_ms": mask_runtime["median"], "runtime_p95_ms": mask_runtime["p95"]},
        {"architecture": "UFLD-v2 ResNet-18", "reference_type": "PROXY_TASK1_SEG2D21", "primary_metric": "lane_f1_tolerant", "value": _aggregate(image_rows, "lane_f1")["mean"], "samples": len(image_rows), "runtime_median_ms": lane_runtime["median"], "runtime_p95_ms": lane_runtime["p95"]},
        {"architecture": "Calibrated IPM", "reference_type": "SELF_GEOMETRY", "primary_metric": "roundtrip_p95_px", "value": float(np.mean([value["roundtrip_pixel_p95"] for value in geometry.values()])), "samples": len(geometry), "runtime_mean_ms_per_frame": projection_summary.get("latency_ms_mean_new")},
        {"architecture": "Six-camera fusion", "reference_type": "PROXY_TASK1_GT_VEC", "primary_metric": "bev_drive_iou", "value": _aggregate(bev_rows, "drive_iou")["mean"], "samples": len(bev_rows), "runtime_mean_ms_per_frame": projection_summary.get("latency_ms_mean_new")},
        {"architecture": "Six-camera fusion", "reference_type": "PROXY_TASK1_GT_VEC", "primary_metric": "bev_lane_chamfer_m", "value": _aggregate(bev_rows, "lane_chamfer_m")["mean"], "samples": len(bev_rows), "runtime_mean_ms_per_frame": projection_summary.get("latency_ms_mean_new")},
    ]
    _write_csv(output_dir / "metrics" / "architecture_scorecard.csv", scorecard)
    _chart(output_dir / "metrics" / "charts" / "road_iou_by_camera.png", "Road IoU by camera [PROXY]", list(CAMERAS), [_aggregate([row for row in image_rows if row["camera"] == camera], "iou")["mean"] or 0 for camera in CAMERAS])
    _chart(output_dir / "metrics" / "charts" / "lane_f1_by_camera.png", "Lane F1 by camera [PROXY]", list(CAMERAS), [_aggregate([row for row in image_rows if row["camera"] == camera], "lane_f1")["mean"] or 0 for camera in CAMERAS], color=(40, 220, 220))
    summary = {
        "status": "PASS_PROXY_ONLY",
        "warning": "No human-audited GT supplied. Scores measure agreement with Task 1 pseudo-labels and self-consistency, not absolute accuracy.",
        "image_records": len(image_rows),
        "bev_records": len(bev_rows),
        "mask2former_road_iou_proxy": _aggregate(image_rows, "iou"),
        "ufld_lane_f1_proxy": _aggregate(image_rows, "lane_f1"),
        "bev_drive_iou_proxy": _aggregate(bev_rows, "drive_iou"),
        "bev_lane_chamfer_m_proxy": _aggregate(bev_rows, "lane_chamfer_m"),
        "cross_camera_drive_iou_self": _aggregate(bev_rows, "cross_camera_drive_iou_self"),
        "temporal_drive_iou_self": _aggregate(bev_rows, "temporal_drive_iou_self"),
        "runtime": {"mask2former_ms": mask_runtime, "ufld_v2_ms": lane_runtime, "projection_mean_ms_per_frame": projection_summary.get("latency_ms_mean_new")},
    }
    atomic_json(output_dir / "metrics" / "summary.json", summary)
    return summary
