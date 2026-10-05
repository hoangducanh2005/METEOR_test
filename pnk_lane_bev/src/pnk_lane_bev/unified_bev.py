"""Calibrated IPM and confidence-weighted six-camera fusion for unified masks."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .constants import CAMERAS
from .geometry import build_camera_geometry, remap_to_bev
from .io import atomic_json, atomic_npz, load_task1_manifest
from .project import bev_stem
from .unified_config import load_unified_config


def run_projection(
    scene_dir: Path,
    output_dir: Path,
    config_path: Path,
    force: bool = False,
    limit: int | None = None,
) -> dict:
    config = load_unified_config(config_path)
    manifest = load_task1_manifest(scene_dir)
    frames = manifest["frames"][:limit] if limit is not None else manifest["frames"]
    geometries = {camera: build_camera_geometry(camera, manifest["cams"][camera]) for camera in CAMERAS}
    atomic_json(
        output_dir / "metrics" / "geometry.json",
        {camera: geometry.metrics for camera, geometry in geometries.items()},
    )
    output_root = output_dir / "bev"
    visual_root = output_dir / "bev_visual" / "per_camera"
    elapsed_values = []
    completed = 0
    for frame_index, frame in enumerate(frames, 1):
        stem = bev_stem(frame)
        target = output_root / f"{stem}.npz"
        if target.exists() and not force:
            completed += 1
            continue
        started = time.perf_counter()
        drives, lanes, valids, fusion_weights, lane_contributions = [], [], [], [], []
        for camera in CAMERAS:
            prediction_path = (
                output_dir / "cache" / "mask2former_unified" / camera /
                f"{int(frame['frame']):04d}.npz"
            )
            if not prediction_path.exists():
                raise FileNotFoundError(prediction_path)
            with np.load(prediction_path) as data:
                drive_probability = data["drivable_probability"].astype(np.float32)
                lane_probability = data["lane_probability_post"].astype(np.float32)
                semantic_confidence = data["semantic_confidence"].astype(np.float32)
            geometry = geometries[camera]
            drive_bev = remap_to_bev(drive_probability, geometry)
            lane_bev = remap_to_bev(lane_probability, geometry)
            semantic_confidence_bev = remap_to_bev(semantic_confidence, geometry)
            camera_config = config["cameras"][camera]
            prior = float(camera_config["reliability_weight"])
            if not bool(camera_config.get("enabled", True)):
                prior = 0.0
            confidence_floor = float(config["fusion"]["semantic_confidence_floor"])
            weight = geometry.weight * prior * np.maximum(semantic_confidence_bev, confidence_floor)
            weight[~geometry.valid] = 0
            drives.append(drive_bev)
            lanes.append(lane_bev)
            valids.append((geometry.valid & (prior > 0)).astype(np.uint8))
            fusion_weights.append(weight)
            lane_contributions.append(weight * lane_bev)

        drive_array = np.stack(drives)
        lane_array = np.stack(lanes)
        valid_array = np.stack(valids)
        weight_array = np.stack(fusion_weights)
        lane_numerator = np.stack(lane_contributions)
        drive_denominator = weight_array.sum(axis=0)
        observed = drive_denominator > 1e-6
        fused_drive = (drive_array * weight_array).sum(axis=0) / np.maximum(drive_denominator, 1e-6)
        lane_active = lane_array > 0.05
        lane_weights = weight_array * lane_active
        lane_denominator = lane_weights.sum(axis=0)
        fused_lane = (lane_array * lane_weights).sum(axis=0) / np.maximum(lane_denominator, 1e-6)
        fused_drive[~observed] = 0
        fused_lane[lane_denominator <= 1e-6] = 0
        lane_share = lane_numerator / np.maximum(lane_numerator.sum(axis=0, keepdims=True), 1e-6)
        drive_threshold = float(config["fusion"]["drivable_threshold"])
        lane_threshold = float(config["fusion"]["lane_threshold"])
        drive_mask = ((fused_drive >= drive_threshold) & observed).astype(np.uint8)
        lane_mask = ((fused_lane >= lane_threshold) & observed).astype(np.uint8)
        coverage = valid_array.sum(axis=0).astype(np.uint8)
        atomic_npz(
            target,
            drivable_probability=fused_drive.astype(np.float16),
            drivable_mask=drive_mask,
            lane_probability=fused_lane.astype(np.float16),
            lane_mask=lane_mask,
            coverage=coverage,
            per_camera_drivable=drive_array.astype(np.float16),
            per_camera_lane=lane_array.astype(np.float16),
            per_camera_valid=valid_array,
            per_camera_fusion_weight=weight_array.astype(np.float16),
            per_camera_lane_contribution=lane_numerator.astype(np.float16),
            per_camera_lane_contribution_share=lane_share.astype(np.float16),
        )
        output_root.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_root / f"{stem}_drivable.png"), np.uint8(np.clip(fused_drive, 0, 1) * 255))
        cv2.imwrite(str(output_root / f"{stem}_lane.png"), np.uint8(np.clip(fused_lane, 0, 1) * 255))
        for camera_index, camera in enumerate(CAMERAS):
            camera_dir = visual_root / camera
            camera_dir.mkdir(parents=True, exist_ok=True)
            canvas = np.zeros((*drive_mask.shape, 3), dtype=np.uint8)
            canvas[drive_array[camera_index] >= drive_threshold] = (45, 155, 55)
            canvas[lane_array[camera_index] >= lane_threshold] = (0, 240, 255)
            cv2.imwrite(str(camera_dir / f"{stem}.png"), canvas)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        elapsed_values.append(elapsed_ms)
        completed += 1
        print(f"[mask2former-ipm] {frame_index}/{len(frames)} {stem} {elapsed_ms:.1f}ms", flush=True)
    summary = {
        "frames": len(frames), "completed": completed,
        "fusion": "sum(camera_prior * semantic_confidence * distance_reliability * M_c) / sum(weights)",
        "drivable_threshold": float(config["fusion"]["drivable_threshold"]),
        "lane_threshold": float(config["fusion"]["lane_threshold"]),
        "camera_weights": {
            camera: float(config["cameras"][camera]["reliability_weight"])
            if config["cameras"][camera].get("enabled", True) else 0.0
            for camera in CAMERAS
        },
        "latency_ms_mean_new": float(np.mean(elapsed_values)) if elapsed_values else None,
    }
    atomic_json(output_dir / "projection_summary.json", summary)
    return summary
