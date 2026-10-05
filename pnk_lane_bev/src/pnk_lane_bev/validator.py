"""Acceptance gate for Task 2 artifacts."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .constants import (
    BEV_HEIGHT, BEV_WIDTH, CAMERAS, CANVAS_HEIGHT, CANVAS_WIDTH,
    MASK2FORMER_MODEL, UFLD_SOURCE_COMMIT, UFLD_SOURCE_REPOSITORY,
)
from .io import atomic_json, load_task1_manifest, read_json
from .project import bev_stem


def validate(scene_dir: Path, output_dir: Path, limit: int | None = None) -> dict:
    manifest = load_task1_manifest(scene_dir)
    frames = manifest["frames"][:limit] if limit is not None else manifest["frames"]
    errors, warnings = [], []
    for frame in frames:
        index = f"{int(frame['frame']):04d}"
        for camera in CAMERAS:
            for path in (
                output_dir / "cache" / "mask2former_small" / camera / f"{index}.npz",
                output_dir / "cache" / "mask2former_small" / camera / f"{index}.json",
                output_dir / "cache" / "ufld_v2" / camera / f"{index}.json",
            ):
                if not path.exists():
                    errors.append(f"missing: {path}")
        stem = bev_stem(frame)
        bev = output_dir / "bev" / f"{stem}.npz"
        visual = output_dir / "frames" / f"{stem}.png"
        if not bev.exists():
            errors.append(f"missing: {bev}")
        else:
            with np.load(bev) as data:
                for key in ("drivable_mask", "lane_mask", "coverage"):
                    if key not in data or data[key].shape != (BEV_HEIGHT, BEV_WIDTH):
                        errors.append(f"bad {key}: {bev}")
        if not visual.exists():
            errors.append(f"missing: {visual}")
        else:
            image = cv2.imread(str(visual), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (CANVAS_HEIGHT, CANVAS_WIDTH):
                errors.append(f"bad visual: {visual}")
    geometry = read_json(output_dir / "metrics" / "geometry.json")
    for camera, metrics in geometry.items():
        if metrics["roundtrip_pixel_p95"] >= 2.0:
            errors.append(f"{camera} roundtrip p95 >= 2 px")
        if metrics["valid_coverage_ratio"] <= 0:
            errors.append(f"{camera} has zero BEV coverage")
    video_path = output_dir / "task2_6cam_bev.mp4"
    capture = cv2.VideoCapture(str(video_path))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if capture.isOpened() else 0
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if capture.isOpened() else 0
    capture.release()
    if video_frames != len(frames):
        errors.append(f"video frame count {video_frames} != {len(frames)}")
    if (video_width, video_height) != (CANVAS_WIDTH, CANVAS_HEIGHT):
        errors.append(f"video size {(video_width, video_height)} != {(CANVAS_WIDTH, CANVAS_HEIGHT)}")
    metrics_summary = read_json(output_dir / "metrics" / "summary.json")
    if metrics_summary.get("status") != "PASS_PROXY_ONLY":
        errors.append("metrics status must be PASS_PROXY_ONLY without audit GT")
    mask_summary = read_json(output_dir / "mask2former_summary.json")
    ufld_summary = read_json(output_dir / "ufld_summary.json")
    report = {
        "status": "PASS_PROXY_ONLY" if not errors else "FAIL",
        "reference_type": "PROXY_AND_SELF_CONSISTENCY",
        "warning": "No human-audited GT supplied; proxy agreement is not absolute accuracy.",
        "counts": {"frames": len(frames), "camera_images": len(frames) * len(CAMERAS), "video_frames": video_frames},
        "video": {"path": str(video_path.resolve()), "width": video_width, "height": video_height},
        "model_provenance": {
            "drivable": {"model": MASK2FORMER_MODEL, "revision": mask_summary.get("model_revision")},
            "lane": {
                "model": "UFLD-v2 CULane ResNet-18",
                "source": UFLD_SOURCE_REPOSITORY,
                "source_commit": UFLD_SOURCE_COMMIT,
                "checkpoint_sha256": ufld_summary.get("checkpoint_sha256"),
            },
        },
        "stage_summaries": {
            "mask2former": mask_summary,
            "ufld_v2": ufld_summary,
            "projection": read_json(output_dir / "projection_summary.json"),
            "render": read_json(output_dir / "render_summary.json"),
        },
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics_summary,
    }
    atomic_json(output_dir / "report.json", report)
    run_manifest_path = output_dir / "run_manifest.json"
    run_manifest = read_json(run_manifest_path)
    run_manifest["status"] = report["status"]
    run_manifest["report"] = str((output_dir / "report.json").resolve())
    atomic_json(run_manifest_path, run_manifest)
    if errors:
        raise RuntimeError(f"Task 2 validation failed with {len(errors)} errors; see report.json")
    return report
