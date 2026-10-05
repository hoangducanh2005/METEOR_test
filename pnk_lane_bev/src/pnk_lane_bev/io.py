"""Input contract, atomic writes, and run manifest handling."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .constants import CAMERAS, IMAGE_HEIGHT, IMAGE_WIDTH, MASK2FORMER_MODEL


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_task1_manifest(scene_dir: Path) -> dict:
    manifest_path = scene_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Task 1 manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    missing_cameras = [camera for camera in CAMERAS if camera not in manifest.get("cams", {})]
    if missing_cameras:
        raise ValueError(f"Task 1 manifest is missing cameras: {missing_cameras}")
    if tuple(manifest.get("img_hw", [])) != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(
            f"Expected canonical images {IMAGE_HEIGHT}x{IMAGE_WIDTH}, got {manifest.get('img_hw')}"
        )
    if not manifest.get("frames"):
        raise ValueError("Task 1 manifest contains no frames")
    return manifest


def image_path(scene_dir: Path, frame: dict, camera: str) -> Path:
    return scene_dir / frame["imgs"][camera]


def read_image(scene_dir: Path, frame: dict, camera: str) -> np.ndarray:
    path = image_path(scene_dir, frame, camera)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    if image.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(f"Unexpected image shape {image.shape} for {path}")
    return image


def prepare_run(scene_dir: Path, output_dir: Path) -> dict:
    task1 = load_task1_manifest(scene_dir)
    missing = []
    for frame in task1["frames"]:
        for camera in CAMERAS:
            path = image_path(scene_dir, frame, camera)
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} camera images; first={missing[0]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for relative in (
        "cache/mask2former_small",
        "cache/ufld_v2",
        "bev",
        "bev_visual",
        "frames",
        "metrics/charts",
        "evidence/image_space",
        "evidence/bev",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "schema_version": 1,
        "task": "drivable_lane_detection_ipm",
        "scene": task1["scene"],
        "task1_scene_dir": str(scene_dir.resolve()),
        "task1_manifest": str((scene_dir / "manifest.json").resolve()),
        "reference_type": "task1_pseudo_gt",
        "status": "PREPARED",
        "models": {
            "drivable": MASK2FORMER_MODEL,
            "lane": "UFLD-v2 CULane ResNet-18",
        },
        "cameras": list(CAMERAS),
        "image_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "bev": {
            "height": int(task1["bev_h"]),
            "width": int(task1["bev_w"]),
            "resolution": float(task1["bev_res"]),
            "forward": float(task1["bev_xh"]),
            "half_width": float(task1["bev_yh"]),
        },
        "frames": [
            {
                "frame": int(frame["frame"]),
                "timestamp": int(frame["timestamp"]),
                "sample_token": frame.get("sample_token"),
                "images": {camera: frame["imgs"][camera] for camera in CAMERAS},
            }
            for frame in task1["frames"]
        ],
        "counts": {"frames": len(task1["frames"]), "camera_images": len(task1["frames"]) * len(CAMERAS)},
    }
    atomic_json(output_dir / "run_manifest.json", run_manifest)
    return run_manifest
