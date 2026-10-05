"""Functional acceptance checks for the Mask2Former-only inference path."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .constants import CAMERAS, IMAGE_HEIGHT, IMAGE_WIDTH
from .io import atomic_json, load_task1_manifest, read_json
from .project import bev_stem


REQUIRED_IMAGE_ARRAYS = {
    "panoptic", "semantic", "semantic_confidence", "drivable_mask", "drivable_probability",
    "lane_mask_raw", "lane_mask_post", "lane_probability_raw", "lane_probability_post",
}
REQUIRED_BEV_ARRAYS = {
    "drivable_probability", "drivable_mask", "lane_probability", "lane_mask", "coverage",
    "per_camera_drivable", "per_camera_lane", "per_camera_valid",
    "per_camera_lane_contribution_share",
}


def validate(scene_dir: Path, output_dir: Path, limit: int | None = None) -> dict:
    manifest = load_task1_manifest(scene_dir)
    frames = manifest["frames"][:limit] if limit is not None else manifest["frames"]
    failures = []
    image_count = 0
    artifact_sets = 0
    for frame in frames:
        for camera in CAMERAS:
            stem = f"{int(frame['frame']):04d}"
            npz_path = output_dir / "cache" / "mask2former_unified" / camera / f"{stem}.npz"
            json_path = npz_path.with_suffix(".json")
            if not npz_path.exists() or not json_path.exists():
                failures.append(f"missing prediction {camera}/{stem}")
                continue
            with np.load(npz_path) as data:
                missing = REQUIRED_IMAGE_ARRAYS - set(data.files)
                if missing:
                    failures.append(f"missing arrays {camera}/{stem}: {sorted(missing)}")
                if data["semantic"].shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                    failures.append(f"bad semantic shape {camera}/{stem}: {data['semantic'].shape}")
            record = read_json(json_path)
            if record.get("model") != "facebook/mask2former-swin-large-mapillary-vistas-panoptic":
                failures.append(f"unexpected model {camera}/{stem}: {record.get('model')}")
            if "ufld" in str(record).lower():
                failures.append(f"UFLD reference in unified inference record {camera}/{stem}")
            image_count += 1
            artifact_root = output_dir / "artifacts" / "per_camera" / camera
            required_artifacts = [
                artifact_root / f"{bev_stem(frame)}_{suffix}.{extension}"
                for suffix, extension in (
                    ("rgb", "jpg"), ("semantic", "png"), ("drivable", "png"),
                    ("lane_raw", "png"), ("lane_post", "png"), ("bev", "png"), ("debug", "jpg"),
                )
            ]
            if all(path.exists() for path in required_artifacts):
                artifact_sets += 1
            else:
                failures.append(f"missing artifact set {camera}/{stem}")
    bev_count = 0
    fused_visual_count = 0
    for frame in frames:
        stem = bev_stem(frame)
        path = output_dir / "bev" / f"{stem}.npz"
        if not path.exists():
            failures.append(f"missing BEV {stem}")
            continue
        with np.load(path) as data:
            missing = REQUIRED_BEV_ARRAYS - set(data.files)
            if missing:
                failures.append(f"missing BEV arrays {stem}: {sorted(missing)}")
            if data["per_camera_lane"].shape[0] != len(CAMERAS):
                failures.append(f"BEV does not contain six cameras: {stem}")
        bev_count += 1
        visual = output_dir / "artifacts" / "fused_bev" / f"{stem}.png"
        if visual.exists() and cv2.imread(str(visual)) is not None:
            fused_visual_count += 1
        else:
            failures.append(f"missing fused visualization {stem}")
    video = output_dir / "deliverables" / "task2_mask2former_unified_6cam_bev.mp4"
    if not video.exists() or video.stat().st_size == 0:
        failures.append("missing deliverable video")
    summary_path = output_dir / "metrics" / "summary.json"
    if not summary_path.exists():
        failures.append("missing metrics summary")
    report = {
        "status": "PASS_PROXY_ONLY" if not failures else "FAIL",
        "inference_path": "Mask2Former only; UFLD-v2 absent",
        "expected_camera_images": len(frames) * len(CAMERAS),
        "camera_images": image_count, "camera_artifact_sets": artifact_sets,
        "expected_bev_frames": len(frames), "bev_frames": bev_count,
        "fused_visual_frames": fused_visual_count, "video": str(video.resolve()),
        "failures": failures,
    }
    atomic_json(output_dir / "validation_report.json", report)
    return report
