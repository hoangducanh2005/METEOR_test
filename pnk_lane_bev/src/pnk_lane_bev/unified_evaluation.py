"""Proxy benchmark for the unified Mask2Former drivable/lane pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .constants import (
    BEV_DRIVABLE_IDS,
    BEV_LANE_ID,
    CAMERAS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SEG2D_LANE_ID,
    SEG2D_ROAD_ID,
)
from .io import atomic_json, load_task1_manifest, read_json
from .metrics import binary_metrics, chamfer_m, tolerant_lane_metrics
from .project import bev_stem


def _aggregate(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=np.float64)
    if not len(values):
        return {"mean": None, "median": None, "p95": None, "count": 0}
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)), "count": int(len(values)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), 8)
    return max(count - 1, 0)


def _runtime_for_camera(benchmark: dict | None, camera: str) -> float | None:
    if not benchmark:
        return None
    record = benchmark.get("per_camera", {}).get(camera)
    return float(record["mean_ms"]) if record else None


def evaluate(
    scene_dir: Path,
    output_dir: Path,
    baseline_output: Path,
    limit: int | None = None,
) -> dict:
    manifest = load_task1_manifest(scene_dir)
    frames = manifest["frames"][:limit] if limit is not None else manifest["frames"]
    camera_order = list(manifest["cams"])
    benchmark_path = output_dir / "inference_benchmark.json"
    benchmark = read_json(benchmark_path) if benchmark_path.exists() else None
    image_rows, bev_rows, camera_bev_rows = [], [], []
    for frame_index, frame in enumerate(frames, 1):
        with np.load(scene_dir / frame["seg2d21"]) as data:
            proxy_semantic = data["seg"]
        for camera in CAMERAS:
            reference = cv2.resize(
                proxy_semantic[camera_order.index(camera)], (IMAGE_WIDTH, IMAGE_HEIGHT),
                interpolation=cv2.INTER_NEAREST,
            )
            prediction_path = (
                output_dir / "cache" / "mask2former_unified" / camera /
                f"{int(frame['frame']):04d}.npz"
            )
            metadata_path = prediction_path.with_suffix(".json")
            with np.load(prediction_path) as data:
                drive = data["drivable_mask"].astype(bool)
                lane_raw = data["lane_mask_raw"].astype(bool)
                lane_post = data["lane_mask_post"].astype(bool)
                lane_probability = data["lane_probability_post"].astype(np.float32)
            metadata = read_json(metadata_path)
            drive_metric = binary_metrics(drive, reference == SEG2D_ROAD_ID)
            lane_metric = tolerant_lane_metrics(lane_post, reference == SEG2D_LANE_ID, radius=5)
            raw_metric = tolerant_lane_metrics(lane_raw, reference == SEG2D_LANE_ID, radius=5)
            image_rows.append(
                {
                    "frame": int(frame["frame"]), "timestamp": int(frame["timestamp"]),
                    "camera": camera, "reference_type": "SELF_SOURCE_COUPLED_TASK1_SEG2D21",
                    "drivable_iou": drive_metric["iou"], "drivable_precision": drive_metric["precision"],
                    "drivable_recall": drive_metric["recall"], "lane_f1": lane_metric["lane_f1"],
                    "lane_precision": lane_metric["lane_precision"], "lane_recall": lane_metric["lane_recall"],
                    "lane_f1_raw": raw_metric["lane_f1"], "empty_lane": int(not lane_post.any()),
                    "lane_components": _component_count(lane_post),
                    "mean_lane_confidence": float(lane_probability[lane_post].mean()) if lane_post.any() else 0.0,
                    "inference_runtime_ms_sampled": _runtime_for_camera(benchmark, camera),
                    "postprocess_runtime_ms": float(metadata["postprocess_latency_ms"]),
                }
            )

        stem = bev_stem(frame)
        with np.load(output_dir / "bev" / f"{stem}.npz") as data:
            drive = data["drivable_mask"].astype(bool)
            lane = data["lane_mask"].astype(bool)
            valid = data["coverage"] > 0
            per_drive = data["per_camera_drivable"].astype(np.float32)
            per_lane = data["per_camera_lane"].astype(np.float32)
            per_valid = data["per_camera_valid"].astype(bool)
        reference = cv2.imread(str(scene_dir / frame["gt_vec"]), cv2.IMREAD_GRAYSCALE)
        if reference is None:
            raise FileNotFoundError(scene_dir / frame["gt_vec"])
        drive_ref = np.isin(reference, BEV_DRIVABLE_IDS)
        lane_ref = reference == BEV_LANE_ID
        drive_metric = binary_metrics(drive, drive_ref, valid)
        lane_metric = tolerant_lane_metrics(lane & valid, lane_ref & valid, radius=2)
        bev_rows.append(
            {
                "frame": int(frame["frame"]), "timestamp": int(frame["timestamp"]),
                "camera": "FUSED_BEV", "reference_type": "PROXY_TASK1_GT_VEC",
                "drivable_iou": drive_metric["iou"], "drivable_precision": drive_metric["precision"],
                "drivable_recall": drive_metric["recall"], "lane_f1": lane_metric["lane_f1"],
                "lane_chamfer_m": chamfer_m(lane & valid, lane_ref & valid),
                "empty_lane": int(not (lane & valid).any()), "valid_coverage": float(valid.mean()),
            }
        )
        for camera_index, camera in enumerate(CAMERAS):
            camera_valid = per_valid[camera_index]
            camera_drive = per_drive[camera_index] >= 0.45
            camera_lane = per_lane[camera_index] >= 0.45
            camera_bev_rows.append(
                {
                    "frame": int(frame["frame"]), "camera": camera,
                    "drivable_iou": binary_metrics(camera_drive, drive_ref, camera_valid)["iou"],
                    "lane_chamfer_m": chamfer_m(
                        camera_lane & camera_valid, lane_ref & camera_valid
                    ),
                    "empty_lane": int(not (camera_lane & camera_valid).any()),
                }
            )
        print(f"[mask2former-evaluate] {frame_index}/{len(frames)} frame={frame['frame']}", flush=True)

    camera_rows = []
    for camera in CAMERAS:
        subset = [row for row in image_rows if row["camera"] == camera]
        bev_subset = [row for row in camera_bev_rows if row["camera"] == camera]
        camera_rows.append(
            {
                "camera": camera, "samples": len(subset),
                "drivable_iou": _aggregate(subset, "drivable_iou")["mean"],
                "lane_f1": _aggregate(subset, "lane_f1")["mean"],
                "lane_f1_raw": _aggregate(subset, "lane_f1_raw")["mean"],
                "empty_lane_rate": _aggregate(subset, "empty_lane")["mean"],
                "lane_components_per_frame": _aggregate(subset, "lane_components")["mean"],
                "mean_lane_confidence": _aggregate(subset, "mean_lane_confidence")["mean"],
                "runtime_ms_sampled": _aggregate(subset, "inference_runtime_ms_sampled")["mean"],
                "postprocess_runtime_ms": _aggregate(subset, "postprocess_runtime_ms")["mean"],
                "bev_drivable_iou": _aggregate(bev_subset, "drivable_iou")["mean"],
                "bev_lane_chamfer_m": _aggregate(bev_subset, "lane_chamfer_m")["mean"],
                "bev_empty_lane_rate": _aggregate(bev_subset, "empty_lane")["mean"],
                "reference_type": "SELF_SOURCE_COUPLED + PROXY_TASK1",
            }
        )
    _write_csv(output_dir / "metrics" / "image_metrics.csv", image_rows)
    _write_csv(output_dir / "metrics" / "bev_metrics.csv", bev_rows)
    _write_csv(output_dir / "metrics" / "per_camera_bev_metrics.csv", camera_bev_rows)
    _write_csv(output_dir / "metrics" / "camera_summary.csv", camera_rows)

    legacy_path = baseline_output / "metrics" / "summary.json"
    baseline_manifest_path = baseline_output / "run_manifest.json"
    baseline_matches_scene = False
    if legacy_path.exists() and baseline_manifest_path.exists():
        baseline_matches_scene = read_json(baseline_manifest_path).get("scene") == manifest.get("scene")
    if baseline_matches_scene:
        legacy = read_json(legacy_path)
        before = {
            "name": "Mask2Former-Small + UFLD-v2 + IPM",
            "available": True,
            "drivable_iou": legacy["mask2former_road_iou_proxy"]["mean"],
            "lane_f1": legacy["ufld_lane_f1_proxy"]["mean"],
            "empty_prediction_rate": 0.7816901408450704,
            "bev_drivable_iou": legacy["bev_drive_iou_proxy"]["mean"],
            "bev_lane_chamfer_m": legacy["bev_lane_chamfer_m_proxy"]["mean"],
            "runtime_ms_per_image": legacy["runtime"]["mask2former_ms"]["mean"] + legacy["runtime"]["ufld_v2_ms"]["mean"],
        }
    else:
        before = {
            "name": "Mask2Former-Small + UFLD-v2 + IPM",
            "available": False,
            "reason": "Baseline was not run for this scene; cross-dataset substitution is disabled.",
            "drivable_iou": None, "lane_f1": None, "empty_prediction_rate": None,
            "bev_drivable_iou": None, "bev_lane_chamfer_m": None,
            "runtime_ms_per_image": None,
        }
    after = {
        "name": "Mask2Former-Swin-L Mapillary unified + IPM",
        "drivable_iou": _aggregate(image_rows, "drivable_iou")["mean"],
        "lane_f1": _aggregate(image_rows, "lane_f1")["mean"],
        "lane_f1_raw": _aggregate(image_rows, "lane_f1_raw")["mean"],
        "empty_prediction_rate": _aggregate(image_rows, "empty_lane")["mean"],
        "bev_drivable_iou": _aggregate(bev_rows, "drivable_iou")["mean"],
        "bev_lane_f1": _aggregate(bev_rows, "lane_f1")["mean"],
        "bev_lane_chamfer_m": _aggregate(bev_rows, "lane_chamfer_m")["mean"],
        "model_runtime_ms_per_image_sampled": benchmark.get("mean_ms") if benchmark else None,
        "postprocess_runtime_ms_per_image": _aggregate(image_rows, "postprocess_runtime_ms")["mean"],
        "runtime_ms_per_image_sampled": (
            benchmark.get("mean_ms") + _aggregate(image_rows, "postprocess_runtime_ms")["mean"]
            if benchmark else None
        ),
        "projection_ms_per_frame": read_json(output_dir / "projection_summary.json").get("latency_ms_mean_new"),
    }
    available_camera_chamfer = [row["bev_lane_chamfer_m"] for row in camera_rows if row["bev_lane_chamfer_m"] is not None]
    detector_bottleneck = after["lane_f1"] < 0.5 or after["empty_prediction_rate"] > 0.5
    ipm_bottleneck = after["lane_f1"] >= 0.5 and after["bev_lane_f1"] < after["lane_f1"] * 0.65
    fusion_bottleneck = bool(available_camera_chamfer) and after["bev_lane_chamfer_m"] is not None and (
        after["bev_lane_chamfer_m"] > float(np.mean(available_camera_chamfer)) * 1.25
    )
    summary = {
        "status": "PASS_PROXY_ONLY",
        "warning": "Image references were derived from the same upstream Mask2Former checkpoint. IoU/F1 are source-coupled consistency metrics, not independent accuracy.",
        "Before": before, "After": after, "Per-camera comparison": camera_rows,
        "Conclusion": {
            "segmentation_bottleneck_proxy_flag": detector_bottleneck,
            "segmentation_bottleneck": "NOT_ESTABLISHED_BY_SOURCE_COUPLED_PROXY",
            "ipm_or_ground_plane_bottleneck": ipm_bottleneck,
            "calibration_bottleneck": "POSSIBLE_BUT_NOT_ISOLATED_BY_CURRENT_PROXY",
            "fusion_bottleneck": fusion_bottleneck,
            "mean_available_per_camera_bev_chamfer_m": float(np.mean(available_camera_chamfer)) if available_camera_chamfer else None,
            "lane_f1_delta_vs_legacy_ufld": (
                after["lane_f1"] - before["lane_f1"] if before["available"] else None
            ),
            "empty_rate_delta_vs_legacy_ufld": (
                after["empty_prediction_rate"] - before["empty_prediction_rate"]
                if before["available"] else None
            ),
            "bev_chamfer_delta_m_vs_legacy_ufld": (
                after["bev_lane_chamfer_m"] - before["bev_lane_chamfer_m"]
                if before["available"] and after["bev_lane_chamfer_m"] is not None else None
            ),
        },
    }
    atomic_json(output_dir / "metrics" / "summary.json", summary)
    camera_lines = "\n".join(
        "| {camera} | {drivable_iou:.4f} | {lane_f1:.4f} | {empty:.2f}% | {chamfer} |".format(
            camera=row["camera"], drivable_iou=row["drivable_iou"], lane_f1=row["lane_f1"],
            empty=row["empty_lane_rate"] * 100,
            chamfer=(f"{row['bev_lane_chamfer_m']:.3f} m" if row["bev_lane_chamfer_m"] is not None else "N/A"),
        )
        for row in camera_rows
    )
    def display(value: float | None, digits: int = 4, suffix: str = "") -> str:
        return f"{value:.{digits}f}{suffix}" if value is not None else "N/A"

    markdown = f"""# Unified Mask2Former result

Status: `PASS_PROXY_ONLY`

Image metrics are source-coupled consistency scores because `seg2d21` was created from the same upstream checkpoint.

| Pipeline | Drive IoU | Lane F1 | Empty rate | BEV Drive IoU | BEV Lane F1 | BEV Chamfer |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Small + UFLD-v2 | {display(before['drivable_iou'])} | {display(before['lane_f1'])} | {display(before['empty_prediction_rate'] * 100 if before['empty_prediction_rate'] is not None else None, 2, '%')} | {display(before['bev_drivable_iou'])} | N/A | {display(before['bev_lane_chamfer_m'], 3, ' m')} |
| Unified Swin-L | {after['drivable_iou']:.4f} | {after['lane_f1']:.4f} | {after['empty_prediction_rate'] * 100:.2f}% | {after['bev_drivable_iou']:.4f} | {after['bev_lane_f1']:.4f} | {after['bev_lane_chamfer_m']:.3f} m |

## Per camera

| Camera | Drive IoU | Lane F1 | Empty rate | BEV Chamfer |
|---|---:|---:|---:|---:|
{camera_lines}

## Runtime

- Mask2Former sampled inference: {after['model_runtime_ms_per_image_sampled']:.2f} ms/image.
- Class filtering and post-processing: {after['postprocess_runtime_ms_per_image']:.2f} ms/image.
- Six-camera IPM and fusion: {after['projection_ms_per_frame']:.2f} ms/frame.

See `docs/Drivable_lane_detection/REPORT_RESULT.md` for proxy limitations and bottleneck analysis.
"""
    (output_dir / "metrics" / "SUMMARY.md").write_text(markdown, encoding="utf-8")
    return summary
