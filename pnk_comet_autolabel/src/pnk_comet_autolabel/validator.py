"""Contract checks for the T4 scene and mandatory 2D segmentation."""

from __future__ import annotations

import base64
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as cocomask

from .constants import CAMERA_ORDER


def validate_scene(scene_dir: Path, report_path: Path | None = None) -> dict:
    ann_dir = scene_dir / "annotation"
    load = lambda name: json.loads((ann_dir / f"{name}.json").read_text(encoding="utf-8"))
    categories = {row["token"]: row["name"] for row in load("category")}
    samples = load("sample")
    sample_tokens = {row["token"] for row in samples}
    sample_data = load("sample_data")
    sd_tokens = {row["token"] for row in sample_data}
    by_sample = defaultdict(set)
    for row in sample_data:
        if row["sample_token"] not in sample_tokens:
            raise ValueError(f"sample_data points to unknown sample: {row['token']}")
        if row["is_key_frame"]:
            by_sample[row["sample_token"]].add(row["filename"].split("/")[1])
        if not (scene_dir / row["filename"]).exists():
            raise FileNotFoundError(scene_dir / row["filename"])
    required = set(CAMERA_ORDER) | {"LIDAR_CONCAT"}
    bad_samples = {key: sorted(required - value) for key, value in by_sample.items()
                   if required - value}
    if bad_samples:
        raise ValueError(f"Samples have missing channels: {bad_samples}")

    camera_sds = {row["token"] for row in sample_data if row.get("height") and row.get("width")}
    anns_by_sd = defaultdict(int)
    class_counts = Counter()
    decoded_pixels = 0
    for table in ("surface_ann", "object_ann"):
        for row in load(table):
            if row["sample_data_token"] not in sd_tokens:
                raise ValueError(f"{table} points to unknown sample_data: {row['token']}")
            if row["category_token"] not in categories:
                raise ValueError(f"{table} points to unknown category: {row['token']}")
            rle = row.get("mask")
            if not rle:
                raise ValueError(f"{table} has no mask: {row['token']}")
            decoded = cocomask.decode({"size": rle["size"],
                                       "counts": base64.b64decode(rle["counts"])})
            if decoded.shape != (432, 768) or not decoded.any():
                raise ValueError(f"Invalid mask for {row['token']}: {decoded.shape}")
            decoded_pixels += int(decoded.sum())
            anns_by_sd[row["sample_data_token"]] += 1
            class_counts[categories[row["category_token"]]] += 1
    missing_seg = sorted(camera_sds - set(anns_by_sd))
    if missing_seg:
        raise RuntimeError(
            f"FAIL: missing segmentation for {len(missing_seg)} camera sample_data tokens; "
            f"first={missing_seg[:5]}"
        )

    instances = {row["token"] for row in load("instance")}
    boxes = load("sample_annotation")
    for row in boxes:
        if row["sample_token"] not in sample_tokens or row["instance_token"] not in instances:
            raise ValueError(f"Broken sample_annotation reference: {row['token']}")
        values = np.asarray(row["translation"] + row["size"] + row["rotation"], dtype=float)
        if not np.isfinite(values).all() or np.any(np.asarray(row["size"]) <= 0):
            raise ValueError(f"Invalid 3D box values: {row['token']}")

    report = {
        "status": "PASS",
        "scene": scene_dir.name,
        "sample_count": len(samples),
        "sample_data_count": len(sample_data),
        "segmented_camera_views": len(anns_by_sd),
        "required_camera_views": len(camera_sds),
        "segmentation_annotations": int(sum(class_counts.values())),
        "segmentation_pixels_decoded": decoded_pixels,
        "segmentation_categories": dict(sorted(class_counts.items())),
        "instances_3d": len(instances),
        "sample_annotations_3d": len(boxes),
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def validate_factory_segmentation(bevlane_scene: Path, report_path: Path | None = None) -> dict:
    manifest = json.loads((bevlane_scene / "manifest.json").read_text(encoding="utf-8"))
    errors = []
    class_pixels = Counter()
    for frame in manifest["frames"]:
        relative = frame.get("seg2d21")
        if not relative:
            errors.append(f"frame={frame['frame']}: missing seg2d21 key")
            continue
        path = bevlane_scene / relative
        try:
            seg = np.load(path)["seg"]
        except Exception as exc:
            errors.append(f"frame={frame['frame']}: cannot decode {path}: {exc}")
            continue
        if seg.shape != (8, 108, 192):
            errors.append(f"frame={frame['frame']}: shape={seg.shape}")
            continue
        for camera_index, channel in enumerate(CAMERA_ORDER):
            view = seg[camera_index]
            if np.all(view == 255):
                errors.append(f"frame={frame['frame']} camera={channel}: all pixels ignored")
            valid = view[view != 255]
            if valid.size and (valid.min() < 0 or valid.max() > 20):
                errors.append(f"frame={frame['frame']} camera={channel}: class outside 0..20")
            for value, count in zip(*np.unique(valid, return_counts=True)):
                class_pixels[int(value)] += int(count)
    report = {
        "status": "FAIL" if errors else "PASS",
        "scene": bevlane_scene.name,
        "frames": len(manifest["frames"]),
        "segmented_views": len(manifest["frames"]) * 8 - len(errors),
        "class_pixels": dict(sorted(class_pixels.items())),
        "errors": errors,
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("FAIL: missing seg2d21\n" + "\n".join(errors[:20]))
    return report

