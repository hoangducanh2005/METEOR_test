"""Artifact and non-regression validation for lane branch v2."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .constants import CAMERAS
from .io import atomic_json, load_task1_manifest, read_json
from .lane_pipeline import DETECTORS
from .project import bev_stem


def _csv_count(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def validate(
    scene_dir: Path,
    output_dir: Path,
    detectors: tuple[str, ...] = DETECTORS,
    limit: int | None = None,
) -> dict:
    manifest = load_task1_manifest(scene_dir)
    frames = manifest["frames"][:limit] if limit is not None else manifest["frames"]
    expected_images = len(frames) * len(CAMERAS)
    failures: list[str] = []
    counts = {}
    drivable_exact_matches = 0
    for detector in detectors:
        cache_count = 0
        bev_count = 0
        for frame in frames:
            for camera in CAMERAS:
                record_path = output_dir / "lane_v2" / detector / "cache" / camera / f"{int(frame['frame']):04d}.json"
                if not record_path.exists():
                    failures.append(f"missing lane cache: {record_path}")
                    continue
                record = read_json(record_path)
                required = {"detector", "lanes", "mean_confidence", "valid", "runtime_ms", "preprocess"}
                if not required.issubset(record):
                    failures.append(f"invalid lane cache schema: {record_path}")
                cache_count += 1
            stem = bev_stem(frame)
            baseline_path = output_dir / "bev" / f"{stem}.npz"
            v2_path = output_dir / "lane_v2" / detector / "bev" / f"{stem}.npz"
            if not v2_path.exists():
                failures.append(f"missing BEV: {v2_path}")
                continue
            with np.load(baseline_path) as baseline, np.load(v2_path) as v2:
                required_arrays = {
                    "lane_probability",
                    "lane_mask",
                    "per_camera_lane",
                    "per_camera_contribution",
                    "per_camera_contribution_share",
                    "drivable_probability",
                    "drivable_mask",
                }
                if not required_arrays.issubset(v2.files):
                    failures.append(f"missing BEV arrays: {v2_path}")
                for key in ("lane_probability", "per_camera_contribution"):
                    if key in v2 and not np.isfinite(v2[key]).all():
                        failures.append(f"non-finite {key}: {v2_path}")
                unchanged = all(
                    np.array_equal(baseline[key], v2[key])
                    for key in ("drivable_probability", "drivable_mask", "coverage")
                )
                if unchanged:
                    drivable_exact_matches += 1
                else:
                    failures.append(f"drivable branch changed: {v2_path}")
            bev_count += 1
        video = output_dir / "lane_v2" / detector / "deliverables" / f"task2_{detector}_6cam_bev.mp4"
        capture = cv2.VideoCapture(str(video))
        video_ok = capture.isOpened()
        video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if video_ok else 0
        video_wh = [
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ] if video_ok else [0, 0]
        capture.release()
        if not video_ok or video_frames != len(frames) or video_wh != [2560, 1440]:
            failures.append(f"invalid video {video}: open={video_ok}, frames={video_frames}, wh={video_wh}")
        rendered = len(list((output_dir / "lane_v2" / detector / "deliverables" / "frames").glob("*.jpg")))
        if cache_count != expected_images or bev_count != len(frames) or rendered != len(frames):
            failures.append(
                f"count mismatch {detector}: cache={cache_count}, bev={bev_count}, rendered={rendered}"
            )
        counts[detector] = {
            "lane_records": cache_count,
            "bev_records": bev_count,
            "rendered_frames": rendered,
            "video_frames": video_frames,
            "video_wh": video_wh,
        }
    preprocess_count = len(list((output_dir / "lane_v2" / "debug" / "preprocess").glob("*/*.jpg")))
    if preprocess_count != expected_images:
        failures.append(f"preprocess visualization count={preprocess_count}, expected={expected_images}")
    metrics_dir = output_dir / "lane_v2" / "metrics"
    metric_counts = {
        "frame_camera_metrics": _csv_count(metrics_dir / "frame_camera_metrics.csv"),
        "frame_bev_metrics": _csv_count(metrics_dir / "frame_bev_metrics.csv"),
        "camera_summary": _csv_count(metrics_dir / "camera_summary.csv"),
    }
    if metric_counts["frame_camera_metrics"] != expected_images * len(detectors):
        failures.append(f"frame-camera metric count mismatch: {metric_counts}")
    if metric_counts["frame_bev_metrics"] != len(frames) * len(detectors):
        failures.append(f"frame-BEV metric count mismatch: {metric_counts}")
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_PROXY_ONLY" if not failures else "FAIL",
        "reference_type": "PROXY_TASK1",
        "frames": len(frames),
        "camera_images": expected_images,
        "detectors": list(detectors),
        "counts": counts,
        "preprocess_visualizations": preprocess_count,
        "metric_counts": metric_counts,
        "drivable_branch_exact_match_records": drivable_exact_matches,
        "drivable_branch_expected_records": len(frames) * len(detectors),
        "metrics_summary": read_json(metrics_dir / "summary.json"),
        "fusion_summary": read_json(output_dir / "lane_v2" / "fusion_summary.json"),
        "failures": failures,
    }
    atomic_json(output_dir / "lane_v2" / "report.json", report)
    if failures:
        raise RuntimeError(f"Lane v2 validation failed with {len(failures)} issue(s); see report.json")
    return report

