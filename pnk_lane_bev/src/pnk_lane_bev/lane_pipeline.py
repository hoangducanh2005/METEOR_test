"""Modular lane inference and confidence-weighted multi-camera BEV fusion."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .clrnet import CLRNetDetector
from .constants import BEV_HEIGHT, BEV_WIDTH, CAMERAS, IMAGE_HEIGHT, IMAGE_WIDTH
from .geometry import bev_ground_grid, build_camera_geometry, remap_to_bev
from .io import atomic_json, atomic_npz, load_task1_manifest, read_json, sha256_file
from .lane_config import load_lane_config
from .lane_preprocess import CameraPreprocessor
from .lane_types import LanePrediction
from .project import bev_stem
from .ufld_v2 import UFLDDetector


DETECTORS = ("ufld_v2", "clrnet", "ufld_v2_res34")


def _make_detector(name: str, checkpoint: Path, device: str):
    if name in ("ufld_v2", "ufld_v2_res34"):
        backbone = "resnet34" if name == "ufld_v2_res34" else "resnet18"
        return UFLDDetector(checkpoint, device=device, backbone=backbone, detector_name=name)
    if name == "clrnet":
        return CLRNetDetector(checkpoint, device=device)
    raise ValueError(f"Unsupported lane detector: {name}")


def _fit_curve(points: list[list[float]]) -> list[float]:
    if len(points) < 3:
        return []
    values = np.asarray(points, dtype=np.float64)
    return [float(value) for value in np.polyfit(values[:, 1], values[:, 0], min(2, len(points) - 1))]


def _canonical_record(
    prediction: LanePrediction,
    preprocessed,
    camera: str,
    frame: dict,
    threshold: float,
    camera_config: dict,
    checkpoint: Path,
    checkpoint_digest: str,
) -> dict:
    raw_lanes = prediction.lanes
    lanes = []
    for lane in raw_lanes:
        if not lane.valid or lane.confidence < threshold or len(lane.points) < 2:
            continue
        points = preprocessed.roi_to_canonical(lane.points)
        points = [point for point in points if 0 <= point[0] < IMAGE_WIDTH and 0 <= point[1] <= IMAGE_HEIGHT]
        if len(points) < 2:
            continue
        lanes.append(
            {
                "points": [[round(value, 3) for value in point] for point in points],
                "curve": _fit_curve(points),
                "confidence": float(lane.confidence),
                "valid": True,
                "metadata": lane.metadata,
            }
        )
    return {
        "schema_version": 2,
        "frame": int(frame["frame"]),
        "timestamp": int(frame["timestamp"]),
        "camera": camera,
        "detector": prediction.detector,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "runtime_ms": float(prediction.runtime_ms),
        "raw_lane_count": len(raw_lanes),
        "lane_count": len(lanes),
        "raw_mean_confidence": float(prediction.confidence),
        "mean_confidence": float(np.mean([lane["confidence"] for lane in lanes])) if lanes else 0.0,
        "confidence_threshold": float(threshold),
        "valid": bool(lanes),
        "no_lane_detected": not lanes,
        "lanes": lanes,
        "camera_fusion": {
            "enabled": bool(camera_config["enabled"]),
            "reliability_weight": float(camera_config["reliability_weight"]),
        },
        "preprocess": {
            "source_path": preprocessed.source_path,
            "undistortion": True,
            "intrinsic_source": preprocessed.intrinsic_source,
            "distortion_camera": preprocessed.distortion_source,
            "roi_xyxy_canonical": list(preprocessed.roi_xyxy),
            "canonical_image_wh": [IMAGE_WIDTH, IMAGE_HEIGHT],
        },
    }


def _save_lane_overlay(path: Path, image: np.ndarray, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = image.copy()
    for index, lane in enumerate(record["lanes"]):
        points = np.rint(np.asarray(lane["points"], np.float32)).astype(np.int32)
        color = ((37 + 71 * index) % 255, 255, (210 - 43 * index) % 255)
        cv2.polylines(canvas, [points[:, None, :]], False, color, 4, cv2.LINE_AA)
        if len(points):
            cv2.putText(
                canvas,
                f"{lane['confidence']:.2f}",
                tuple(points[-1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    status = f"{record['detector']} lanes={record['lane_count']}/{record['raw_lane_count']} threshold={record['confidence_threshold']:.2f}"
    cv2.rectangle(canvas, (0, 0), (IMAGE_WIDTH, 42), (18, 18, 18), -1)
    cv2.putText(canvas, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def run_inference(
    scene_dir: Path,
    output_dir: Path,
    allroot_manifest: Path,
    camera_config_path: Path,
    checkpoints: dict[str, Path],
    detectors: tuple[str, ...] = DETECTORS,
    device: str = "auto",
    force: bool = False,
    limit: int | None = None,
) -> dict:
    manifest = load_task1_manifest(scene_dir)
    frames = manifest["frames"][:limit] if limit is not None else manifest["frames"]
    config = load_lane_config(camera_config_path)
    root = output_dir / "lane_v2"
    preprocessor = CameraPreprocessor(allroot_manifest, config, root / "debug" / "preprocess")
    runners = {name: _make_detector(name, checkpoints[name], device) for name in detectors}
    checkpoint_digests = {name: sha256_file(checkpoints[name]) for name in detectors}
    counters = {name: {"records": 0, "empty": 0, "lane_count": 0, "runtime": []} for name in detectors}
    total = len(frames) * len(CAMERAS)
    for job_index, (frame, camera) in enumerate(
        ((frame, camera) for frame in frames for camera in CAMERAS), 1
    ):
        targets = {
            name: root / name / "cache" / camera / f"{int(frame['frame']):04d}.json"
            for name in detectors
        }
        pending = [name for name in detectors if force or not targets[name].exists()]
        processed = None
        if pending:
            processed = preprocessor.process(
                int(frame["timestamp"]), camera, int(frame["frame"]), save_debug=True
            )
        for name in detectors:
            target = targets[name]
            if name in pending:
                prediction = runners[name].predict(processed.roi_image)
                threshold = float(config["cameras"][camera]["confidence_threshold"][name])
                record = _canonical_record(
                    prediction,
                    processed,
                    camera,
                    frame,
                    threshold,
                    config["cameras"][camera],
                    checkpoints[name],
                    checkpoint_digests[name],
                )
                atomic_json(target, record)
                overlay = root / name / "debug" / "prediction" / camera / f"{int(frame['frame']):04d}.jpg"
                _save_lane_overlay(overlay, processed.canonical_image, record)
            else:
                record = read_json(target)
            counters[name]["records"] += 1
            counters[name]["empty"] += int(not record["valid"])
            counters[name]["lane_count"] += int(record["lane_count"])
            counters[name]["runtime"].append(float(record["runtime_ms"]))
        print(f"[lane-v2-infer] {job_index}/{total} frame={frame['frame']} camera={camera}", flush=True)
    inference_summary_path = root / "inference_summary.json"
    summary = read_json(inference_summary_path) if inference_summary_path.exists() else {}
    for name, values in counters.items():
        count = values["records"]
        summary[name] = {
            "records": count,
            "empty_prediction_rate": values["empty"] / max(count, 1),
            "average_lanes_per_frame": values["lane_count"] / max(count, 1),
            "runtime_ms_mean": float(np.mean(values["runtime"])),
            "runtime_ms_p95": float(np.percentile(values["runtime"], 95)),
        }
    atomic_json(inference_summary_path, summary)
    return summary


def lane_raster(record: dict) -> np.ndarray:
    raster = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    for lane in record.get("lanes", []):
        if not lane.get("valid", True):
            continue
        points = np.rint(np.asarray(lane["points"], np.float32)).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(
                raster,
                [points[:, None, :]],
                False,
                float(lane["confidence"]),
                4,
                cv2.LINE_AA,
            )
    return raster


def _distance_reliability(config: dict) -> np.ndarray:
    x, y = bev_ground_grid()
    distance = np.sqrt(x * x + y * y)
    near = float(config["near_m"])
    far = float(config["far_m"])
    minimum = float(config["minimum"])
    alpha = np.clip((distance - near) / max(far - near, 1e-6), 0, 1)
    return (1 - alpha * (1 - minimum)).astype(np.float32)


def _spatial_weight(geometry, distance_weight: np.ndarray) -> np.ndarray:
    edge = np.minimum.reduce(
        [
            geometry.map_x / IMAGE_WIDTH,
            (IMAGE_WIDTH - geometry.map_x) / IMAGE_WIDTH,
            geometry.map_y / IMAGE_HEIGHT,
            (IMAGE_HEIGHT - geometry.map_y) / IMAGE_HEIGHT,
        ]
    )
    edge = np.clip(edge * 5.0, 0, 1)
    return geometry.valid.astype(np.float32) * edge.astype(np.float32) * distance_weight


def run_fusion(
    scene_dir: Path,
    output_dir: Path,
    camera_config_path: Path,
    detectors: tuple[str, ...] = DETECTORS,
    force: bool = False,
    limit: int | None = None,
) -> dict:
    manifest = load_task1_manifest(scene_dir)
    frames = manifest["frames"][:limit] if limit is not None else manifest["frames"]
    config = load_lane_config(camera_config_path)
    geometries = {camera: build_camera_geometry(camera, manifest["cams"][camera]) for camera in CAMERAS}
    distance = _distance_reliability(config["fusion"]["distance_reliability"])
    summary_path = output_dir / "lane_v2" / "fusion_summary.json"
    summaries = read_json(summary_path) if summary_path.exists() else {}
    for detector in detectors:
        completed = 0
        for frame_index, frame in enumerate(frames, 1):
            stem = bev_stem(frame)
            target = output_dir / "lane_v2" / detector / "bev" / f"{stem}.npz"
            if target.exists() and not force:
                completed += 1
                continue
            per_lane, per_weight, per_valid = [], [], []
            for camera in CAMERAS:
                camera_cfg = config["cameras"][camera]
                record = read_json(
                    output_dir / "lane_v2" / detector / "cache" / camera / f"{int(frame['frame']):04d}.json"
                )
                geometry = geometries[camera]
                image_lane = lane_raster(record)
                bev_lane = remap_to_bev(image_lane, geometry)
                prediction_confidence = float(record["mean_confidence"]) if record["valid"] else 0.0
                prior = float(camera_cfg["reliability_weight"]) if camera_cfg["enabled"] else 0.0
                weight = _spatial_weight(geometry, distance) * prior * prediction_confidence
                weight[bev_lane <= 0] = 0
                per_lane.append(bev_lane)
                per_weight.append(weight)
                per_valid.append(geometry.valid.astype(np.uint8))
            lane_array = np.stack(per_lane).astype(np.float32)
            weight_array = np.stack(per_weight).astype(np.float32)
            numerator = lane_array * weight_array
            denominator = weight_array.sum(axis=0)
            fused = numerator.sum(axis=0) / np.maximum(denominator, 1e-6)
            fused[denominator <= 0] = 0
            contribution_share = numerator / np.maximum(numerator.sum(axis=0, keepdims=True), 1e-6)
            baseline_path = output_dir / "bev" / f"{stem}.npz"
            with np.load(baseline_path) as baseline:
                drivable_probability = baseline["drivable_probability"]
                drivable_mask = baseline["drivable_mask"]
                coverage = baseline["coverage"]
            atomic_npz(
                target,
                drivable_probability=drivable_probability,
                drivable_mask=drivable_mask,
                lane_probability=fused.astype(np.float16),
                lane_mask=(fused >= float(config["fusion"]["lane_threshold"])).astype(np.uint8),
                coverage=coverage,
                per_camera_lane=lane_array.astype(np.float16),
                per_camera_fusion_weight=weight_array.astype(np.float16),
                per_camera_contribution=numerator.astype(np.float16),
                per_camera_contribution_share=contribution_share.astype(np.float16),
                per_camera_valid=np.stack(per_valid),
            )
            visual_dir = target.parent.parent / "bev_visual"
            visual_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(visual_dir / f"{stem}_fused.png"), np.uint8(np.clip(fused, 0, 1) * 255))
            for camera_index, camera in enumerate(CAMERAS):
                camera_dir = visual_dir / "contribution" / camera
                camera_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(camera_dir / f"{stem}.png"),
                    np.uint8(np.clip(contribution_share[camera_index], 0, 1) * 255),
                )
            completed += 1
            print(f"[lane-v2-fusion:{detector}] {frame_index}/{len(frames)}", flush=True)
        summaries[detector] = {
            "frames": completed,
            "formula": "sum(camera_prior * prediction_confidence * distance_reliability * M_c) / sum(active_weights)",
            "camera_weights": {
                camera: float(config["cameras"][camera]["reliability_weight"])
                if config["cameras"][camera]["enabled"]
                else 0.0
                for camera in CAMERAS
            },
        }
    atomic_json(summary_path, summaries)
    return summaries
