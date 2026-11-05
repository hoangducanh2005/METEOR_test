"""Build a T4Dataset scene and export general-purpose annotations."""

from __future__ import annotations

import base64
import json
import math
import uuid
from datetime import date
from pathlib import Path

import cv2
import laspy
import numpy as np
from pycocotools import mask as cocomask

from .calibration import camera_to_ego, load_calibration
from .constants import (CAMERA_MAP, OBJECT_CATEGORIES, REFERENCE_3D_CLASS_MAP,
                        SEGMENTATION_CATEGORIES)
from .geometry import rotate_xy, rotation_matrix_to_quaternion, yaw_quaternion
from .reader import PoseInterpolator, timestamp_ns

NS = uuid.UUID("95804413-301c-4f2e-a37c-b356f53a2b6c")
TARGET_W, TARGET_H = 768, 432


def token(*parts: object) -> str:
    return uuid.uuid5(NS, ":".join(map(str, parts))).hex


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _process_image(source: Path, target: Path, intrinsic: dict) -> list[list[float]]:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {source}")
    k = np.asarray(intrinsic["camera_matrix"], dtype=np.float64)
    dist = np.asarray(intrinsic["distortion_coefficients"], dtype=np.float64)
    image = cv2.undistort(image, k, dist, None, k)
    old_h, old_w = image.shape[:2]
    image = cv2.resize(image, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    scaled = k.copy()
    scaled[0, :] *= TARGET_W / old_w
    scaled[1, :] *= TARGET_H / old_h
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Cannot write image: {target}")
    return scaled.tolist()


def _convert_lidar(source: Path, target: Path) -> int:
    cloud = laspy.read(source)
    intensity = np.asarray(cloud.intensity, dtype=np.float32)
    points = np.column_stack([
        np.asarray(cloud.x, dtype=np.float32),
        np.asarray(cloud.y, dtype=np.float32),
        np.asarray(cloud.z, dtype=np.float32),
        intensity,
        np.zeros(len(cloud.points), dtype=np.float32),
    ]).astype(np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    points.tofile(target)
    return len(points)


def prepare_scene(raw_root: Path, preflight_path: Path, calibration_dir: Path,
                  label_dir: Path, scene_dir: Path, scene_name: str,
                  vehicle: str = "VF6_01", recording_date: date = date(2026, 3, 1),
                  limit: int | None = None) -> dict:
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    samples_in = preflight["samples"][:limit]
    if not samples_in:
        raise ValueError("Preflight contains no accepted samples")
    intrinsics, extrinsics, period_id = load_calibration(calibration_dir, vehicle, recording_date)
    pose = PoseInterpolator(raw_root, samples_in[0]["timestamp_ns"], samples_in[-1]["timestamp_ns"])
    ann_dir = scene_dir / "annotation"
    ann_dir.mkdir(parents=True, exist_ok=True)

    sensors = []
    calibrated = []
    cal_by_channel = {}
    for source, channel in CAMERA_MAP.items():
        sensor_token = token(scene_name, "sensor", channel)
        cal_token = token(scene_name, "calibration", channel)
        sensors.append({"token": sensor_token, "channel": channel, "modality": "camera"})
        t_cam_ego = camera_to_ego(extrinsics[source])
        k = np.asarray(intrinsics[source]["camera_matrix"], dtype=np.float64)
        k[0, :] *= TARGET_W / intrinsics[source]["image_width"]
        k[1, :] *= TARGET_H / intrinsics[source]["image_height"]
        calibrated.append({
            "token": cal_token,
            "sensor_token": sensor_token,
            "translation": t_cam_ego[:3, 3].tolist(),
            "rotation": rotation_matrix_to_quaternion(t_cam_ego[:3, :3]),
            "camera_intrinsic": k.tolist(),
            "camera_distortion": [0.0] * len(intrinsics[source]["distortion_coefficients"]),
        })
        cal_by_channel[channel] = cal_token
    lidar_sensor = token(scene_name, "sensor", "LIDAR_CONCAT")
    lidar_cal = token(scene_name, "calibration", "LIDAR_CONCAT")
    sensors.append({"token": lidar_sensor, "channel": "LIDAR_CONCAT", "modality": "lidar"})
    calibrated.append({
        "token": lidar_cal,
        "sensor_token": lidar_sensor,
        "translation": [0.0, 0.0, 0.0],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "camera_intrinsic": [],
        "camera_distortion": [],
    })
    cal_by_channel["LIDAR_CONCAT"] = lidar_cal

    sample_rows, sample_data, ego_poses, run_samples = [], [], [], []
    point_counts = []
    for index, raw_sample in enumerate(samples_in):
        stamp = int(raw_sample["timestamp_ns"])
        sample_token = token(scene_name, "sample", stamp)
        prev_token = token(scene_name, "sample", samples_in[index - 1]["timestamp_ns"]) if index else ""
        next_token = token(scene_name, "sample", samples_in[index + 1]["timestamp_ns"]) if index + 1 < len(samples_in) else ""
        sample_rows.append({
            "token": sample_token,
            "prev": prev_token,
            "next": next_token,
            "timestamp": stamp // 1000,
            "scene_token": token(scene_name, "scene"),
        })
        run_cameras = {}
        for channel, camera in raw_sample["cameras"].items():
            source = camera["source_channel"]
            image_stamp = int(camera["timestamp_ns"])
            relative = Path("data") / channel / f"{image_stamp}.jpg"
            target = scene_dir / relative
            if not target.exists():
                _process_image(Path(camera["source_path"]), target, intrinsics[source])
            xyz, yaw = pose.at(image_stamp)
            ego_token = token(scene_name, "ego", channel, image_stamp)
            ego_poses.append({"token": ego_token, "translation": xyz, "rotation": yaw_quaternion(yaw),
                              "timestamp": image_stamp // 1000})
            sd_token = token(scene_name, "sample_data", channel, image_stamp)
            sample_data.append({
                "token": sd_token,
                "sample_token": sample_token,
                "ego_pose_token": ego_token,
                "calibrated_sensor_token": cal_by_channel[channel],
                "filename": relative.as_posix(),
                "fileformat": "jpg",
                "is_key_frame": True,
                "height": TARGET_H,
                "width": TARGET_W,
                "timestamp": image_stamp // 1000,
            })
            run_cameras[channel] = {
                "source_channel": source,
                "source_path": camera["source_path"],
                "source_timestamp_ns": image_stamp,
                "delta_ms": camera["delta_ms"],
                "sample_data_token": sd_token,
                "filename": relative.as_posix(),
            }
        lidar_relative = Path("data") / "LIDAR_CONCAT" / f"{stamp}.bin"
        lidar_target = scene_dir / lidar_relative
        if not lidar_target.exists():
            count = _convert_lidar(Path(raw_sample["lidar_path"]), lidar_target)
        else:
            count = lidar_target.stat().st_size // (5 * 4)
        point_counts.append(count)
        xyz, yaw = pose.at(stamp)
        ego_token = token(scene_name, "ego", "LIDAR_CONCAT", stamp)
        ego_poses.append({"token": ego_token, "translation": xyz, "rotation": yaw_quaternion(yaw),
                          "timestamp": stamp // 1000})
        lidar_sd = token(scene_name, "sample_data", "LIDAR_CONCAT", stamp)
        sample_data.append({
            "token": lidar_sd,
            "sample_token": sample_token,
            "ego_pose_token": ego_token,
            "calibrated_sensor_token": lidar_cal,
            "filename": lidar_relative.as_posix(),
            "fileformat": "bin",
            "is_key_frame": True,
            "height": 0,
            "width": 0,
            "timestamp": stamp // 1000,
        })
        run_samples.append({
            "index": index,
            "timestamp_ns": stamp,
            "sample_token": sample_token,
            "lidar": {"source_path": raw_sample["lidar_path"], "filename": lidar_relative.as_posix(),
                      "sample_data_token": lidar_sd, "point_count": count},
            "reference_label_path": str(label_dir / (Path(raw_sample["lidar_path"]).stem + ".txt")),
            "cameras": run_cameras,
        })

    category_names = sorted(set(SEGMENTATION_CATEGORIES) | set(REFERENCE_3D_CLASS_MAP.values()))
    categories = [{"token": token("category", name), "name": name, "description": name}
                  for name in category_names]
    dump(ann_dir / "category.json", categories)
    dump(ann_dir / "sensor.json", sensors)
    dump(ann_dir / "calibrated_sensor.json", calibrated)
    dump(ann_dir / "ego_pose.json", ego_poses)
    dump(ann_dir / "sample.json", sample_rows)
    dump(ann_dir / "sample_data.json", sample_data)
    for name in ("surface_ann", "object_ann", "instance", "sample_annotation"):
        dump(ann_dir / f"{name}.json", [])
    dump(ann_dir / "scene.json", [{
        "token": token(scene_name, "scene"), "name": scene_name,
        "first_sample_token": sample_rows[0]["token"],
        "last_sample_token": sample_rows[-1]["token"], "nbr_samples": len(sample_rows),
    }])
    manifest = {
        "schema_version": 1,
        "scene_name": scene_name,
        "raw_root": str(raw_root.resolve()),
        "calibration_dir": str(calibration_dir.resolve()),
        "calibration_period": period_id,
        "reference_label_dir": str(label_dir.resolve()),
        "image_transform": {"undistort": True, "resize": [TARGET_W, TARGET_H]},
        "camera_slot_note": {
            "CAM_FRONT_NARROW": "PNK CAM_P_L side camera placed in fixed METEOR slot",
            "CAM_BACK_NARROW": "PNK CAM_P_R side camera placed in fixed METEOR slot",
        },
        "pose": pose.stats,
        "samples": run_samples,
    }
    dump(scene_dir / "run_manifest.json", manifest)
    return {
        "scene": scene_name,
        "samples": len(run_samples),
        "camera_images": len(run_samples) * len(CAMERA_MAP),
        "lidar_points_min": min(point_counts),
        "lidar_points_max": max(point_counts),
        "scene_dir": str(scene_dir),
    }


def _rle(mask: np.ndarray) -> dict:
    encoded = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {
        "size": [int(v) for v in encoded["size"]],
        "counts": base64.b64encode(encoded["counts"]).decode("ascii"),
    }


def export_panoptic(scene_dir: Path, cache_dir: Path) -> dict:
    manifest = json.loads((scene_dir / "run_manifest.json").read_text(encoding="utf-8"))
    surface, objects = [], []
    counts: dict[str, int] = {}
    missing = []
    for sample in manifest["samples"]:
        for channel, camera in sample["cameras"].items():
            stem = f"{sample['timestamp_ns']}_{channel}"
            npz_path, meta_path = cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"
            if not npz_path.exists() or not meta_path.exists():
                missing.append(stem)
                continue
            panoptic = np.load(npz_path)["panoptic"]
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if panoptic.shape != (TARGET_H, TARGET_W):
                raise ValueError(f"Unexpected panoptic size for {stem}: {panoptic.shape}")
            mapped_here = 0
            for segment in meta["segments"]:
                category = segment["target_category"]
                mask = panoptic == int(segment["segment_id"])
                if not mask.any():
                    continue
                ys, xs = np.where(mask)
                ann = {
                    "token": token("panoptic", stem, segment["segment_id"]),
                    "sample_data_token": camera["sample_data_token"],
                    "category_token": token("category", category),
                    "mask": _rle(mask),
                    "confidence": float(segment["score"]),
                    "source_model_label": segment["source_label"],
                }
                if category in OBJECT_CATEGORIES:
                    ann["bbox"] = [int(xs.min()), int(ys.min()),
                                   int(xs.max() + 1), int(ys.max() + 1)]
                    objects.append(ann)
                else:
                    surface.append(ann)
                counts[category] = counts.get(category, 0) + 1
                mapped_here += 1
            if mapped_here == 0:
                missing.append(stem + ":no-mapped-segments")
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(f"Segmentation gate failed for {len(missing)} images: {preview}")
    ann_dir = scene_dir / "annotation"
    dump(ann_dir / "surface_ann.json", surface)
    dump(ann_dir / "object_ann.json", objects)
    return {"surface_annotations": len(surface), "object_annotations": len(objects),
            "category_counts": counts}


def export_reference_boxes(scene_dir: Path) -> dict:
    """Temporary 3D bridge for flow testing; labels remain explicit reference provenance.

    Preprocessing note: this bridge currently transforms the original box yaw as
    supplied. It does not consume the separate world-frame heading audit or
    resolve the front/back branch. Keep source labels read-only; see
    docs/DATA_PIPELINE.md before treating exported rotations as directional.
    """
    manifest = json.loads((scene_dir / "run_manifest.json").read_text(encoding="utf-8"))
    instances: dict[str, dict] = {}
    annotations = []
    for sample in manifest["samples"]:
        label_path = Path(sample["reference_label_path"])
        if not label_path.exists():
            raise FileNotFoundError(f"Reference label is missing: {label_path}")
        lidar_sd = sample["lidar"]["sample_data_token"]
        ego_poses = json.loads((scene_dir / "annotation" / "ego_pose.json").read_text(encoding="utf-8"))
        sample_data = json.loads((scene_dir / "annotation" / "sample_data.json").read_text(encoding="utf-8"))
        sd = next(d for d in sample_data if d["token"] == lidar_sd)
        ep = next(e for e in ego_poses if e["token"] == sd["ego_pose_token"])
        qw, _, _, qz = ep["rotation"]
        ego_yaw = 2 * math.atan2(qz, qw)
        for line_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            fields = line.split()
            if len(fields) < 11 or fields[7] not in REFERENCE_3D_CLASS_MAP:
                continue
            x, y, z, length, width, height, box_yaw = map(float, fields[:7])
            category = REFERENCE_3D_CLASS_MAP[fields[7]]
            track_id = fields[10]
            inst_key = f"reference:{category}:{track_id}"
            inst_token = token("instance", inst_key)
            instances.setdefault(inst_key, {
                "token": inst_token,
                "category_token": token("category", category),
                "nbr_annotations": 0,
                "first_annotation_token": "",
                "last_annotation_token": "",
                "source": "reference_3d_bridge",
            })
            dx, dy = rotate_xy(x, y, ego_yaw)
            ann_token = token("reference-box", sample["timestamp_ns"], line_index)
            annotations.append({
                "token": ann_token,
                "sample_token": sample["sample_token"],
                "instance_token": inst_token,
                "translation": [ep["translation"][0] + dx, ep["translation"][1] + dy,
                                ep["translation"][2] + z],
                "size": [width, length, height],
                "rotation": yaw_quaternion(ego_yaw + box_yaw),
                "num_lidar_pts": int(float(fields[9])),
                "source": "reference_3d_bridge_for_e2e_flow_test",
            })
            inst = instances[inst_key]
            inst["nbr_annotations"] += 1
            if not inst["first_annotation_token"]:
                inst["first_annotation_token"] = ann_token
            inst["last_annotation_token"] = ann_token
    dump(scene_dir / "annotation" / "instance.json", list(instances.values()))
    dump(scene_dir / "annotation" / "sample_annotation.json", annotations)
    return {"instances": len(instances), "sample_annotations": len(annotations),
            "source": "reference_3d_bridge", "warning": "Not a CenterPoint prediction or evaluation result"}
