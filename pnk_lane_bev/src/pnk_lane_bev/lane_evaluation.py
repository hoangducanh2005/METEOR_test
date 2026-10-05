"""Detector benchmark and per-camera evidence for the improved lane branch."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .constants import BEV_LANE_ID, CAMERAS, IMAGE_HEIGHT, IMAGE_WIDTH, SEG2D_LANE_ID
from .io import atomic_json, load_task1_manifest, read_json
from .lane_pipeline import DETECTORS, lane_raster
from .metrics import chamfer_m, tolerant_lane_metrics
from .project import bev_stem


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _baseline(scene_dir: Path, output_dir: Path, frames: list[dict], camera_order: list[str]) -> tuple[dict, list[dict]]:
    rows = []
    for frame in frames:
        with np.load(scene_dir / frame["seg2d21"]) as payload:
            semantics = payload["seg"]
        for camera in CAMERAS:
            reference = cv2.resize(
                semantics[camera_order.index(camera)],
                (IMAGE_WIDTH, IMAGE_HEIGHT),
                interpolation=cv2.INTER_NEAREST,
            ) == SEG2D_LANE_ID
            record = read_json(output_dir / "cache" / "ufld_v2" / camera / f"{int(frame['frame']):04d}.json")
            prediction = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.float32)
            for lane in record.get("lanes", []):
                points = np.rint(np.asarray(lane["points"], np.float32)).astype(np.int32)
                if len(points) >= 2:
                    cv2.polylines(prediction, [points[:, None, :]], False, 1.0, 4, cv2.LINE_AA)
            metric = tolerant_lane_metrics(prediction > 0.2, reference, radius=5)
            rows.append(
                {
                    "camera": camera,
                    "lane_f1": metric["lane_f1"],
                    "empty": int(not record.get("lanes")),
                    "lane_count": len(record.get("lanes", [])),
                    "mean_confidence": float(np.mean([lane["confidence"] for lane in record.get("lanes", [])]))
                    if record.get("lanes")
                    else 0.0,
                    "runtime_ms": float(record["latency_ms"]),
                }
            )
    baseline_summary = read_json(output_dir / "metrics" / "summary.json")
    return (
        {
            "name": "before_legacy_ufld_max_fusion",
            "lane_f1": _mean(rows, "lane_f1"),
            "empty_prediction_rate": _mean(rows, "empty"),
            "average_lanes_per_frame": _mean(rows, "lane_count"),
            "runtime_ms": _mean(rows, "runtime_ms"),
            "bev_chamfer_m": baseline_summary["bev_lane_chamfer_m_proxy"]["mean"],
            "reference_type": "PROXY_TASK1",
        },
        rows,
    )


def _detector_rows(scene_dir: Path, output_dir: Path, detector: str, frames: list[dict], camera_order: list[str]):
    image_rows, bev_rows = [], []
    for frame in frames:
        with np.load(scene_dir / frame["seg2d21"]) as payload:
            semantics = payload["seg"]
        stem = bev_stem(frame)
        proxy_bev = cv2.imread(str(scene_dir / frame["gt_vec"]), cv2.IMREAD_GRAYSCALE)
        lane_reference_bev = proxy_bev == BEV_LANE_ID
        with np.load(output_dir / "lane_v2" / detector / "bev" / f"{stem}.npz") as payload:
            fused_lane = payload["lane_mask"].astype(bool)
            coverage = payload["coverage"] > 0
            per_camera = payload["per_camera_lane"].astype(np.float32)
            per_valid = payload["per_camera_valid"].astype(bool)
        fused_metric = tolerant_lane_metrics(fused_lane & coverage, lane_reference_bev & coverage, radius=2)
        bev_rows.append(
            {
                "detector": detector,
                "frame": int(frame["frame"]),
                "camera": "FUSED_BEV",
                "lane_f1": fused_metric["lane_f1"],
                "bev_chamfer_m": chamfer_m(fused_lane & coverage, lane_reference_bev & coverage),
                "reference_type": "PROXY_TASK1_GT_VEC",
            }
        )
        for camera_index, camera in enumerate(CAMERAS):
            reference = cv2.resize(
                semantics[camera_order.index(camera)],
                (IMAGE_WIDTH, IMAGE_HEIGHT),
                interpolation=cv2.INTER_NEAREST,
            ) == SEG2D_LANE_ID
            record = read_json(
                output_dir / "lane_v2" / detector / "cache" / camera / f"{int(frame['frame']):04d}.json"
            )
            prediction = lane_raster(record) >= 0.2
            metric = tolerant_lane_metrics(prediction, reference, radius=5)
            camera_bev = per_camera[camera_index] >= 0.2
            valid = per_valid[camera_index]
            image_rows.append(
                {
                    "detector": detector,
                    "frame": int(frame["frame"]),
                    "timestamp": int(frame["timestamp"]),
                    "camera": camera,
                    "lane_f1": metric["lane_f1"],
                    "lane_precision": metric["lane_precision"],
                    "lane_recall": metric["lane_recall"],
                    "empty": int(not record["valid"]),
                    "lane_count": int(record["lane_count"]),
                    "raw_lane_count": int(record["raw_lane_count"]),
                    "mean_confidence": float(record["mean_confidence"]),
                    "runtime_ms": float(record["runtime_ms"]),
                    "bev_chamfer_m": chamfer_m(camera_bev & valid, lane_reference_bev & valid),
                    "reference_type": "PROXY_TASK1_SEG2D21_AND_GT_VEC",
                }
            )
    return image_rows, bev_rows


def _camera_summary(detector: str, rows: list[dict]) -> list[dict]:
    output = []
    for camera in CAMERAS:
        subset = [row for row in rows if row["camera"] == camera]
        output.append(
            {
                "Detector": detector,
                "Camera": camera,
                "Lane F1": _mean(subset, "lane_f1"),
                "Empty Rate": _mean(subset, "empty"),
                "Lane/frame": _mean(subset, "lane_count"),
                "Mean Confidence": _mean(subset, "mean_confidence"),
                "BEV Chamfer": _mean(subset, "bev_chamfer_m"),
                "Runtime": _mean(subset, "runtime_ms"),
                "Samples": len(subset),
                "Reference": "PROXY_TASK1",
            }
        )
    return output


def _conclusion(before: dict, detector_summaries: dict[str, dict], camera_rows: list[dict]) -> dict:
    best_name = max(detector_summaries, key=lambda name: detector_summaries[name]["lane_f1"] or 0)
    best = detector_summaries[best_name]
    detector_bottleneck = bool((best["lane_f1"] or 0) < 0.30 or best["empty_prediction_rate"] > 0.50)
    valid_camera_chamfers = [row["BEV Chamfer"] for row in camera_rows if row["Detector"] == best_name and row["BEV Chamfer"] is not None]
    camera_chamfer = float(np.mean(valid_camera_chamfers)) if valid_camera_chamfers else None
    ipm_bottleneck = bool(
        (best["lane_f1"] or 0) >= 0.20
        and best["bev_chamfer_m"] is not None
        and best["bev_chamfer_m"] > 5.0
    )
    fusion_bottleneck = bool(
        camera_chamfer is not None
        and best["bev_chamfer_m"] is not None
        and best["bev_chamfer_m"] > camera_chamfer * 1.15
    )
    return {
        "selected_detector_by_lane_f1_proxy": best_name,
        "detector_bottleneck": detector_bottleneck,
        "ipm_bottleneck": ipm_bottleneck,
        "calibration_bottleneck": "POSSIBLE" if ipm_bottleneck else "NOT_ISOLATED_BY_CURRENT_PROXY",
        "fusion_bottleneck": fusion_bottleneck,
        "evidence": {
            "best_lane_f1": best["lane_f1"],
            "best_empty_rate": best["empty_prediction_rate"],
            "best_fused_bev_chamfer_m": best["bev_chamfer_m"],
            "mean_available_per_camera_bev_chamfer_m": camera_chamfer,
            "lane_f1_delta_from_before": (best["lane_f1"] or 0) - (before["lane_f1"] or 0),
        },
        "interpretation": "Proxy agreement only; human-audited lane labels are required for an accuracy claim.",
    }


def _chart(path: Path, camera_rows: list[dict]) -> None:
    width, height = 1250, 620
    canvas = np.full((height, width, 3), 24, np.uint8)
    cv2.putText(canvas, "Lane detector comparison by camera [PROXY]", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2)
    colors = {
        "ufld_v2": (70, 210, 255),
        "clrnet": (100, 230, 120),
        "ufld_v2_res34": (230, 150, 70),
    }
    for camera_index, camera in enumerate(CAMERAS):
        y = 90 + camera_index * 80
        cv2.putText(canvas, camera.replace("CAM_", ""), (20, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        for detector_index, detector in enumerate(DETECTORS):
            row = next(value for value in camera_rows if value["Camera"] == camera and value["Detector"] == detector)
            value = float(row["Lane F1"] or 0)
            y0 = y + detector_index * 27
            cv2.rectangle(canvas, (220, y0), (220 + int(value * 850), y0 + 20), colors[detector], -1)
            cv2.putText(canvas, f"{detector} {value:.3f}", (1080, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[detector], 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def _summary_markdown(summary: dict) -> str:
    before = summary["Before"]
    after = summary["After"]
    rows = summary["Per-camera comparison"]
    lines = [
        "# Lane branch v2 — Before/After",
        "",
        "> Reference: Task 1 pseudo-GT. These are agreement metrics, not human-audited accuracy.",
        "",
        "| Pipeline | Lane F1 | Empty Rate | Lane/frame | BEV Chamfer |",
        "|---|---:|---:|---:|---:|",
        f"| Before: legacy UFLD/max fusion | {before['lane_f1']:.4f} | {before['empty_prediction_rate']:.2%} | {before['average_lanes_per_frame']:.3f} | {before['bev_chamfer_m']:.3f} m |",
    ]
    detector_names = [detector for detector in DETECTORS if detector in after]
    for detector in detector_names:
        value = after[detector]
        lines.append(
            f"| After: {detector} | {value['lane_f1']:.4f} | {value['empty_prediction_rate']:.2%} | "
            f"{value['average_lanes_per_frame']:.3f} | {value['bev_chamfer_m']:.3f} m |"
        )
    lines.extend(
        [
            "",
            "## Per-camera comparison",
            "",
            "| Camera | Before F1 | " + " | ".join(f"{name} F1 | {name} Empty" for name in detector_names) + " |",
            "|---|---:|" + "---:|---:|" * len(detector_names),
        ]
    )
    for camera in CAMERAS:
        values = {
            row["Detector"]: row
            for row in rows
            if row["Camera"] == camera
        }
        cells = [f"{values['before_legacy_ufld']['Lane F1']:.4f}"]
        for detector in detector_names:
            cells.extend([f"{values[detector]['Lane F1']:.4f}", f"{values[detector]['Empty Rate']:.2%}"])
        lines.append(f"| {camera} | " + " | ".join(cells) + " |")
    conclusion = summary["Conclusion"]
    lane_f1_delta = conclusion["evidence"]["lane_f1_delta_from_before"]
    if lane_f1_delta is not None and lane_f1_delta > 0:
        comparison_statement = (
            f"- Best candidate improves Lane F1 by `{lane_f1_delta:+.4f}` over the legacy baseline, "
            "but the absolute score and coverage still require target-domain labeling and fine-tuning."
        )
    else:
        comparison_statement = (
            "- Best candidate does not improve Lane F1 over the legacy baseline; "
            "target-domain labeling and fine-tuning are the next justified step."
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- Selected detector by Lane F1 proxy: **{conclusion['selected_detector_by_lane_f1_proxy']}**.",
            f"- Detector bottleneck: **{conclusion['detector_bottleneck']}**.",
            f"- IPM bottleneck: **{conclusion['ipm_bottleneck']}**.",
            f"- Calibration bottleneck: **{conclusion['calibration_bottleneck']}**.",
            f"- Fusion bottleneck: **{conclusion['fusion_bottleneck']}**.",
            comparison_statement,
            "",
        ]
    )
    return "\n".join(lines)


def evaluate(
    scene_dir: Path,
    output_dir: Path,
    detectors: tuple[str, ...] = DETECTORS,
    limit: int | None = None,
) -> dict:
    manifest = load_task1_manifest(scene_dir)
    frames = manifest["frames"][:limit] if limit is not None else manifest["frames"]
    camera_order = list(manifest["cams"])
    before, baseline_rows = _baseline(scene_dir, output_dir, frames, camera_order)
    detector_summaries, all_image_rows, all_bev_rows, camera_summary = {}, [], [], []
    for detector in detectors:
        image_rows, bev_rows = _detector_rows(scene_dir, output_dir, detector, frames, camera_order)
        all_image_rows.extend(image_rows)
        all_bev_rows.extend(bev_rows)
        camera_summary.extend(_camera_summary(detector, image_rows))
        detector_summaries[detector] = {
            "name": detector,
            "lane_f1": _mean(image_rows, "lane_f1"),
            "empty_prediction_rate": _mean(image_rows, "empty"),
            "average_lanes_per_frame": _mean(image_rows, "lane_count"),
            "mean_confidence": _mean(image_rows, "mean_confidence"),
            "runtime_ms": _mean(image_rows, "runtime_ms"),
            "bev_lane_f1": _mean(bev_rows, "lane_f1"),
            "bev_chamfer_m": _mean(bev_rows, "bev_chamfer_m"),
            "reference_type": "PROXY_TASK1",
        }
    baseline_camera = _camera_summary("before_legacy_ufld", baseline_rows)
    comparison = [before, *detector_summaries.values()]
    conclusion = _conclusion(before, detector_summaries, camera_summary)
    metrics_dir = output_dir / "lane_v2" / "metrics"
    _write_csv(metrics_dir / "frame_camera_metrics.csv", all_image_rows)
    _write_csv(metrics_dir / "frame_bev_metrics.csv", all_bev_rows)
    _write_csv(metrics_dir / "camera_summary.csv", [*baseline_camera, *camera_summary])
    _write_csv(metrics_dir / "detector_comparison.csv", comparison)
    _chart(metrics_dir / "lane_f1_by_camera.png", camera_summary)
    summary = {
        "status": "PASS_PROXY_ONLY",
        "warning": "Metrics measure agreement with Task 1 pseudo-labels, not absolute accuracy.",
        "Before": before,
        "After": detector_summaries,
        "Per-camera comparison": [*baseline_camera, *camera_summary],
        "Conclusion": conclusion,
    }
    atomic_json(metrics_dir / "summary.json", summary)
    (metrics_dir / "SUMMARY.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary
