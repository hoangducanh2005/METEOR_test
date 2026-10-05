"""Project six image-space predictions to BEV and fuse them."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .constants import BEV_DRIVABLE_THRESHOLD, BEV_HEIGHT, BEV_WIDTH, CAMERAS, IMAGE_HEIGHT, IMAGE_WIDTH
from .geometry import build_camera_geometry, remap_to_bev
from .io import atomic_json, atomic_npz, load_task1_manifest, read_json


def bev_stem(frame: dict) -> str:
    return f"{int(frame['frame']):04d}_{int(frame['timestamp'])}"


def lane_raster(record: dict) -> np.ndarray:
    raster = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    for lane in record.get("lanes", []):
        points = np.rint(np.asarray(lane["points"], dtype=np.float32)).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(
                raster,
                [points.reshape(-1, 1, 2)],
                False,
                float(lane.get("confidence", 1.0)),
                thickness=4,
                lineType=cv2.LINE_AA,
            )
    return raster


def run_scene(scene_dir: Path, output_dir: Path, force: bool = False, limit: int | None = None) -> dict:
    manifest = load_task1_manifest(scene_dir)
    frames = manifest["frames"][:limit] if limit is not None else manifest["frames"]
    geometries = {camera: build_camera_geometry(camera, manifest["cams"][camera]) for camera in CAMERAS}
    atomic_json(
        output_dir / "metrics" / "geometry.json",
        {camera: geometry.metrics for camera, geometry in geometries.items()},
    )
    completed, latency = 0, []
    for index, frame in enumerate(frames, 1):
        stem = bev_stem(frame)
        target = output_dir / "bev" / f"{stem}.npz"
        if target.exists() and not force:
            completed += 1
            continue
        started = time.perf_counter()
        camera_drive, camera_lane, camera_valid, camera_weight = [], [], [], []
        for camera in CAMERAS:
            cache_stem = f"{int(frame['frame']):04d}"
            drive_file = output_dir / "cache" / "mask2former_small" / camera / f"{cache_stem}.npz"
            lane_file = output_dir / "cache" / "ufld_v2" / camera / f"{cache_stem}.json"
            if not drive_file.exists() or not lane_file.exists():
                raise FileNotFoundError(f"Missing inference cache for frame={cache_stem}, camera={camera}")
            with np.load(drive_file) as data:
                drive_probability = data["probability"].astype(np.float32)
            lanes = lane_raster(read_json(lane_file))
            geometry = geometries[camera]
            camera_drive.append(remap_to_bev(drive_probability, geometry))
            camera_lane.append(remap_to_bev(lanes, geometry))
            camera_valid.append(geometry.valid.astype(np.uint8))
            camera_weight.append(geometry.weight)

        camera_drive_array = np.stack(camera_drive)
        camera_lane_array = np.stack(camera_lane)
        camera_valid_array = np.stack(camera_valid)
        camera_weight_array = np.stack(camera_weight)
        weight_sum = camera_weight_array.sum(axis=0)
        fused_drive = (camera_drive_array * camera_weight_array).sum(axis=0) / np.maximum(weight_sum, 1e-6)
        fused_lane = (camera_lane_array * camera_weight_array).max(axis=0)
        observed = weight_sum > 1e-6
        fused_drive[~observed] = 0
        fused_lane[~observed] = 0
        coverage = camera_valid_array.sum(axis=0).astype(np.uint8)
        atomic_npz(
            target,
            drivable_probability=fused_drive.astype(np.float16),
            drivable_mask=((fused_drive >= BEV_DRIVABLE_THRESHOLD) & observed).astype(np.uint8),
            lane_probability=fused_lane.astype(np.float16),
            lane_mask=((fused_lane >= 0.20) & observed).astype(np.uint8),
            coverage=coverage,
            per_camera_drivable=camera_drive_array.astype(np.float16),
            per_camera_lane=camera_lane_array.astype(np.float16),
            per_camera_valid=camera_valid_array,
        )
        cv2.imwrite(str(output_dir / "bev" / f"{stem}_drivable.png"), np.uint8(fused_drive * 255))
        cv2.imwrite(str(output_dir / "bev" / f"{stem}_lane.png"), np.uint8(np.clip(fused_lane, 0, 1) * 255))
        cv2.imwrite(str(output_dir / "bev" / f"{stem}_coverage.png"), np.uint8(coverage * 42))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latency.append(elapsed_ms)
        completed += 1
        print(f"[project-bev] {index}/{len(frames)} frame={frame['frame']} {elapsed_ms:.1f}ms", flush=True)
    summary = {
        "frames": len(frames),
        "completed": completed,
        "bev_shape": [BEV_HEIGHT, BEV_WIDTH],
        "drivable_threshold": BEV_DRIVABLE_THRESHOLD,
        "lane_threshold": 0.20,
        "latency_ms_mean_new": float(np.mean(latency)) if latency else None,
    }
    atomic_json(output_dir / "projection_summary.json", summary)
    return summary

