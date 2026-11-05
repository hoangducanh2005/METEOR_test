"""LIDAR_TOP preprocessing and CoMET-replacement handoff for METEOR.

This package deliberately reuses the repository's existing T4Dataset schema,
Mask2Former runner, annotation exporter, and validators. Its new boundary is
the sensor input: LIDAR_TOP is deskewed first, then re-referenced to each
annotation sample time before it is adapted to T4Dataset.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from datetime import date
from pathlib import Path

import laspy
import numpy as np

from pnk_comet_autolabel.constants import CAMERA_MAP, REFERENCE_3D_CLASS_MAP
from pnk_comet_autolabel.exporter import (
    export_panoptic,
    export_reference_boxes,
    prepare_scene,
    token,
)
from pnk_comet_autolabel.geometry import yaw_quaternion
from pnk_comet_autolabel.mask2former import DEFAULT_MODEL, run_scene
from pnk_comet_autolabel.reader import nearest_path, timestamp_ns
from pnk_comet_autolabel.sensor_preprocessing import (
    NavTimeline,
    _camera_files,
    _read_nav,
    preprocess_session,
)
from pnk_comet_autolabel.validator import validate_scene

from .heading import build_session_heading_audit


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run_sensor_preprocessing(raw_root: Path, calibration_dir: Path, output_dir: Path,
                             visualization_stride: int = 10) -> dict:
    """Call the prior preprocessing implementation with a local API shim.

    The existing function passes ``max_nav_gap_ms`` to ``NavTimeline``, whose
    constructor names the same setting ``max_gap_ms``. Keep this compatibility
    fix scoped to the new runner instead of editing the user's existing folder.
    """
    import pnk_comet_autolabel.sensor_preprocessing as sensor_preprocessing

    original = sensor_preprocessing.NavTimeline

    class CompatibleNavTimeline(original):
        def __init__(self, rows, max_gap_ms=50.0, max_nav_gap_ms=None):
            super().__init__(rows, max_gap_ms=(max_gap_ms if max_nav_gap_ms is None
                                               else max_nav_gap_ms))

    sensor_preprocessing.NavTimeline = CompatibleNavTimeline
    try:
        return sensor_preprocessing.preprocess_session(
            raw_root, calibration_dir, output_dir,
            visualization_stride=visualization_stride,
        )
    finally:
        sensor_preprocessing.NavTimeline = original


def _read_freeze_candidates(preprocessed_dir: Path) -> dict[str, set[str]]:
    path = preprocessed_dir / "camera_freeze_candidates.csv"
    result: dict[str, set[str]] = {channel: set() for channel in CAMERA_MAP}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            result.setdefault(row["channel"], set()).add(row["timestamp"])
    return result


def _write_anchor_laz(path: Path, xyz: np.ndarray, intensity: np.ndarray) -> None:
    finite = np.isfinite(xyz).all(axis=1)
    xyz = np.asarray(xyz[finite], dtype=np.float64)
    intensity = np.asarray(intensity[finite], dtype=np.float64)
    if len(xyz) == 0:
        raise ValueError(f"No finite points to serialize to {path}")
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001], dtype=np.float64)
    header.offsets = np.floor(xyz.min(axis=0) / 100.0) * 100.0
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    cloud.intensity = np.clip(np.rint(intensity), 0, 65535).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud.write(path)


def build_top_preflight(
    raw_root: Path,
    label_dir: Path,
    preprocessed_dir: Path,
    preflight_path: Path,
    staging_dir: Path,
    camera_max_delta_ms: float = 35.0,
    top_max_delta_ms: float = 55.0,
    max_nav_gap_ms: float = 50.0,
) -> dict:
    """Match annotated sample times to TOP LiDAR and camera streams.

    Labels define the 5 Hz sample anchors. No LIDAR_MERGED_EGO cloud is read.
    Deskewed TOP clouds are transformed from their scan-end ego frame to the
    exact label timestamp using NAV poses, then staged as canonical ego-frame
    LAZ files for the existing T4 adapter.
    """
    raw_root, label_dir = raw_root.resolve(), label_dir.resolve()
    deskewed_dir = preprocessed_dir / "lidar_top_deskewed_ego"
    if not (preprocessed_dir / "preprocessing_report.json").exists():
        raise FileNotFoundError(f"Run LIDAR_TOP preprocessing first: {preprocessed_dir}")
    labels = sorted(label_dir.glob("*.txt"), key=timestamp_ns)
    top_dir = raw_root / "LIDAR" / "LIDAR_TOP"
    top = sorted(((timestamp_ns(p), p) for p in top_dir.glob("*.laz")), key=lambda x: x[0])
    if not labels or not top:
        raise FileNotFoundError(f"Need label .txt files and LIDAR_TOP .laz files: {label_dir}, {top_dir}")
    cameras = _camera_files(raw_root)
    missing = [source for source in CAMERA_MAP if not cameras.get(source)]
    if missing:
        raise FileNotFoundError(f"Required pinhole camera streams are missing: {missing}")

    anchor_stamps = [timestamp_ns(p) for p in labels]
    nav_rows = _read_nav(raw_root, min(anchor_stamps[0], top[0][0]),
                         max(anchor_stamps[-1], top[-1][0]))
    nav = NavTimeline(nav_rows, max_gap_ms=max_nav_gap_ms)
    frozen = _read_freeze_candidates(preprocessed_dir)
    used_top: set[Path] = set()
    used_cameras: dict[str, set[Path]] = {channel: set() for channel in CAMERA_MAP}
    staged_dir = staging_dir / "lidar_top_at_annotation_time"
    samples, lidar_map = [], []
    all_camera_deltas, top_deltas = [], []

    for label_path, anchor_ns in zip(labels, anchor_stamps):
        top_ns, top_path = nearest_path(top, anchor_ns)
        top_delta_ms = (top_ns - anchor_ns) / 1e6
        if abs(top_delta_ms) > top_max_delta_ms:
            raise RuntimeError(f"TOP scan is {top_delta_ms:.3f} ms from label anchor {label_path.stem}")
        if top_path in used_top:
            raise RuntimeError(f"A LIDAR_TOP scan would be reused for two labels: {top_path}")
        used_top.add(top_path)
        deskewed_path = deskewed_dir / f"{top_path.stem}.npz"
        if not deskewed_path.exists():
            raise FileNotFoundError(f"Deskewed TOP cloud is missing: {deskewed_path}")

        with np.load(deskewed_path) as payload:
            xyz_top = np.asarray(payload["xyz_ego_at_scan_end"], dtype=np.float64)
            intensity = np.asarray(payload["intensity"], dtype=np.float32)
            cloud_stamp = int(payload["scan_end_timestamp_ns"])
        if cloud_stamp != top_ns:
            raise ValueError(f"Deskewed cloud timestamp mismatch for {top_path.name}")

        pose_top = nav.at(np.asarray([top_ns], dtype=np.int64))
        pose_anchor = nav.at(np.asarray([anchor_ns], dtype=np.int64))
        if not bool(pose_top.valid[0] and pose_anchor.valid[0]):
            raise RuntimeError(f"NAV pose unavailable for TOP-to-anchor transform: {label_path.stem}")
        # Row-vector form of EGO(top) -> ENU -> EGO(annotation timestamp).
        xyz_world = xyz_top @ pose_top.rotation_world_ego[0].T + pose_top.position_enu[0]
        xyz_anchor = (xyz_world - pose_anchor.position_enu[0]) @ pose_anchor.rotation_world_ego[0]
        staged_laz = staged_dir / f"{label_path.stem}.laz"
        _write_anchor_laz(staged_laz, xyz_anchor, intensity)

        matched_cameras = {}
        for source, channel in CAMERA_MAP.items():
            camera_ns, image_path = nearest_path(cameras[source], anchor_ns)
            delta_ms = (camera_ns - anchor_ns) / 1e6
            if abs(delta_ms) > camera_max_delta_ms:
                raise RuntimeError(f"{channel} is {delta_ms:.3f} ms from anchor {label_path.stem}")
            if image_path in used_cameras[source]:
                raise RuntimeError(f"Camera image would be reused for {channel}: {image_path.name}")
            used_cameras[source].add(image_path)
            all_camera_deltas.append(abs(delta_ms))
            matched_cameras[channel] = {
                "source_channel": source,
                "source_path": str(image_path),
                "timestamp_ns": camera_ns,
                "delta_ms": abs(delta_ms),
                "signed_delta_ms": delta_ms,
                "sync_status": "suspected_freeze" if image_path.stem in frozen.get(source, set())
                               else ("good_sync" if abs(delta_ms) <= 15.0 else "deskew_required"),
                "usable_for_crossmodal": image_path.stem not in frozen.get(source, set()),
            }

        top_deltas.append(abs(top_delta_ms))
        samples.append({
            "timestamp_ns": anchor_ns,
            "lidar_path": str(staged_laz),
            "top_source_path": str(top_path),
            "top_timestamp_ns": top_ns,
            "top_delta_ms": top_delta_ms,
            "deskewed_source_path": str(deskewed_path),
            "point_cloud_frame": "ego at annotation timestamp",
            "reference_label_path": str(label_path),
            "cameras": matched_cameras,
        })
        lidar_map.append({
            "annotation_timestamp": label_path.stem,
            "annotation_timestamp_ns": anchor_ns,
            "lidar_top_file": top_path.name,
            "lidar_top_timestamp_ns": top_ns,
            "lidar_top_delta_ms": top_delta_ms,
            "deskewed_cloud": str(deskewed_path),
            "canonical_cloud": str(staged_laz),
            "point_count": int(len(xyz_anchor)),
        })

    preprocessed_report = json.loads(
        (preprocessed_dir / "preprocessing_report.json").read_text(encoding="utf-8")
    )
    result = {
        "raw_root": str(raw_root),
        "lidar_source": "LIDAR_TOP",
        "lidar_input_count": len(top),
        "annotation_anchor_count": len(samples),
        "camera_input_counts": {k: len(v) for k, v in cameras.items()},
        "accepted_count": len(samples),
        "dropped_count": 0,
        "sync": {
            "camera_max_delta_ms": max(all_camera_deltas),
            "camera_p95_delta_ms": float(np.percentile(all_camera_deltas, 95)),
            "camera_threshold_ms": camera_max_delta_ms,
            "top_max_delta_ms": max(top_deltas),
            "top_p95_delta_ms": float(np.percentile(top_deltas, 95)),
            "top_threshold_ms": top_max_delta_ms,
        },
        "preprocessing_summary": preprocessed_report["summary"],
        "samples": samples,
        "lidar_anchor_map": lidar_map,
        "raw_data_modified": False,
    }
    _dump(preflight_path, result)
    _dump(staging_dir / "lidar_top_anchor_map.json", lidar_map)
    return result


def build_t4_scene(raw_root: Path, calibration_dir: Path, label_dir: Path,
                   preprocessed_dir: Path, preflight_path: Path,
                   staging_dir: Path, scene_dir: Path, scene_name: str,
                   vehicle: str = "VF6_01") -> dict:
    preflight = build_top_preflight(raw_root, label_dir, preprocessed_dir,
                                    preflight_path, staging_dir)
    match = re_session_date(raw_root)
    result = prepare_scene(raw_root, preflight_path, calibration_dir, label_dir,
                           scene_dir, scene_name, vehicle, match)
    manifest_path = scene_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_stamp = {int(x["timestamp_ns"]): x for x in preflight["samples"]}
    for sample in manifest["samples"]:
        source = by_stamp[int(sample["timestamp_ns"])]
        sample["lidar"].update({
            "source_sensor": "LIDAR_TOP",
            "raw_source_path": source["top_source_path"],
            "raw_source_timestamp_ns": source["top_timestamp_ns"],
            "source_delta_from_annotation_ms": source["top_delta_ms"],
            "deskewed_source_path": source["deskewed_source_path"],
            "point_cloud_frame": source["point_cloud_frame"],
            "reference_label_path": source["reference_label_path"],
        })
        for channel, camera in sample["cameras"].items():
            original = source["cameras"][channel]
            camera["sync_status"] = original["sync_status"]
            camera["usable_for_crossmodal"] = original["usable_for_crossmodal"]
    manifest["sensor_preprocessing"] = {
        "lidar_source": "LIDAR_TOP",
        "deskew": "point_timestamp + NAV position/Heading2/Pitch/Roll to scan end",
        "reference_time": "label timestamp; deskewed cloud rigidly re-referenced with NAV pose",
        "coordinate_frame": "ego forward-left-up at sample timestamp",
        "raw_data_modified": False,
        "camera_health_report": str(preprocessed_dir / "preprocessing_report.json"),
    }
    manifest["annotation_provenance"] = {
        "camera_2d": "Mask2Former Swin-L Mapillary Vistas panoptic prediction",
        "lidar_3d": "supplied PNK 3D annotations; no CenterPoint inference",
        "warning": "3D boxes are supplied labels; heading/yaw is recomputed from NAV-compensated track motion when evidence passes the confidence gates.",
    }
    heading_summary = build_session_heading_audit(
        raw_root, label_dir, scene_dir / "annotation" / "heading_audit")
    manifest["heading_audit"] = heading_summary
    _dump(manifest_path, manifest)
    return {**result, "preflight": str(preflight_path), "scene_dir": str(scene_dir),
            "lidar_source": "LIDAR_TOP", "heading_audit": heading_summary,
            "raw_data_modified": False}


def re_session_date(raw_root: Path) -> date:
    import re
    match = re.match(r"^(\d{4})(\d{2})(\d{2})_", raw_root.name)
    if not match:
        raise ValueError(f"Cannot infer recording date from session name: {raw_root.name}")
    return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_equivalent_mask_cache(scene_dir: Path, target_cache: Path,
                               source_scene: Path, source_cache: Path,
                               model_name: str = DEFAULT_MODEL) -> dict:
    """Reuse prior predictions only when the encoded model inputs match exactly."""
    manifest = json.loads((scene_dir / "run_manifest.json").read_text(encoding="utf-8"))
    old_manifest_path = source_scene / "run_manifest.json"
    old_manifest = (json.loads(old_manifest_path.read_text(encoding="utf-8"))
                    if old_manifest_path.exists() else {"samples": []})
    old_views = {}
    for sample in old_manifest.get("samples", []):
        for channel, camera in sample["cameras"].items():
            old_views[(int(sample["timestamp_ns"]), channel)] = camera
    target_cache.mkdir(parents=True, exist_ok=True)
    copied, skipped = 0, 0
    for sample in manifest["samples"]:
        stamp = int(sample["timestamp_ns"])
        for channel, camera in sample["cameras"].items():
            old = old_views.get((stamp, channel))
            stem = f"{stamp}_{channel}"
            src_npz, src_json = source_cache / f"{stem}.npz", source_cache / f"{stem}.json"
            dst_npz, dst_json = target_cache / f"{stem}.npz", target_cache / f"{stem}.json"
            target_image = scene_dir / camera["filename"]
            if (old is None or not src_npz.exists() or not src_json.exists()
                    or not target_image.exists()):
                skipped += 1
                continue
            old_image = source_scene / old["filename"]
            if not old_image.exists() or _sha256(old_image) != _sha256(target_image):
                skipped += 1
                continue
            metadata = json.loads(src_json.read_text(encoding="utf-8"))
            if metadata.get("model") != model_name:
                skipped += 1
                continue
            if not dst_npz.exists():
                shutil.copy2(src_npz, dst_npz)
            if not dst_json.exists():
                shutil.copy2(src_json, dst_json)
            copied += 1
    return {"exact_input_cache_reused": copied, "not_reused": skipped,
            "model": model_name, "source_cache": str(source_cache)}


def apply_heading_audit(scene_dir: Path, audit_csv: Path,
                        segment: str) -> dict:
    """Apply only evidence-backed world headings; leave unresolved yaw untouched."""
    audit_rows: dict[tuple[str, str, str], dict] = {}
    with audit_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("segment") == segment:
                audit_rows[(row["timestamp"], row["trackid"], row.get("class", ""))] = row

    manifest = json.loads((scene_dir / "run_manifest.json").read_text(encoding="utf-8"))
    annotation_path = scene_dir / "annotation" / "sample_annotation.json"
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    by_token = {ann["token"]: ann for ann in annotations}
    sidecar, source_counts = [], {}
    matched, valid = 0, 0
    for sample in manifest["samples"]:
        label_path = Path(sample["reference_label_path"])
        stamp = label_path.stem
        sample_ns = int(sample["timestamp_ns"])
        for line_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            fields = line.split()
            if len(fields) < 11 or fields[7] not in REFERENCE_3D_CLASS_MAP:
                continue
            ann_token = token("reference-box", sample_ns, line_index)
            ann = by_token.get(ann_token)
            if ann is None:
                continue
            raw_yaw = float(fields[6])
            row = (audit_rows.get((stamp, fields[10], fields[7]))
                   or audit_rows.get((stamp, fields[10], "")))
            is_valid = bool(row and row.get("heading_valid", "").lower() == "true"
                            and row.get("heading_world_deg", "") not in ("", "None"))
            source = row.get("heading_source", "heading_invalid") if row else "not_in_audit"
            confidence = float(row.get("heading_confidence", 0.0) or 0.0) if row else 0.0
            world_deg = float(row["heading_world_deg"]) if is_valid else None
            if is_valid:
                ann["rotation"] = yaw_quaternion(math.radians(world_deg))
                matched += 1
                valid += 1
            source_counts[source] = source_counts.get(source, 0) + 1
            sidecar.append({
                "annotation_token": ann_token,
                "timestamp": stamp,
                "track_id": fields[10],
                "class": fields[7],
                "raw_yaw_rad": raw_yaw,
                "heading_valid": is_valid,
                "heading_source": source if row else "heading_invalid",
                "heading_confidence": confidence,
                "heading_world_deg": world_deg,
                "rotation_policy": "audit world heading" if is_valid else "retain modulo-180 source yaw",
            })
    annotation_path.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    sidecar_path = scene_dir / "annotation" / "annotation_heading_audit.json"
    _dump(sidecar_path, sidecar)
    return {"audit_segment": segment, "audit_rows": len(audit_rows),
            "exported_reference_boxes": len(annotations), "sidecar_rows": len(sidecar),
            "heading_valid_applied": valid, "heading_invalid_or_unmatched": len(sidecar) - valid,
            "source_counts": source_counts, "sidecar": str(sidecar_path),
            "source_labels_modified": False}


def export_general_annotations(scene_dir: Path, cache_dir: Path,
                               heading_audit: Path | None = None,
                               segment: str | None = None) -> dict:
    manifest = json.loads((scene_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if heading_audit is None:
        generated = manifest.get("heading_audit") or {}
        heading_audit = Path(generated.get("resolved_csv", ""))
        if not str(heading_audit) or not heading_audit.is_file():
            raise FileNotFoundError("No run-local heading audit was generated; pass --heading-audit explicitly")
        segment = segment or generated.get("segment")
    if not segment:
        raise ValueError("Heading audit segment is required")
    panoptic = export_panoptic(scene_dir, cache_dir)
    boxes = export_reference_boxes(scene_dir)
    heading = apply_heading_audit(scene_dir, heading_audit, segment)
    return {"panoptic": panoptic, "boxes_3d": boxes, "heading_audit": heading}


def run_mask2former(scene_dir: Path, cache_dir: Path, model: str = DEFAULT_MODEL,
                    device: str = "auto", force: bool = False,
                    reuse_scene: Path | None = None,
                    reuse_cache: Path | None = None) -> dict:
    reuse = {"exact_input_cache_reused": 0, "not_reused": 0}
    if reuse_scene and reuse_cache and not force:
        reuse = seed_equivalent_mask_cache(scene_dir, cache_dir, reuse_scene,
                                           reuse_cache, model)
    result = run_scene(scene_dir, cache_dir, model, device, force=force)
    return {**result, "cache_reuse": reuse,
            "inference_model": "Mask2Former Swin-L Mapillary Vistas Panoptic"}


def write_run_report(output_path: Path, scene_name: str,
                     preprocessing_dir: Path, preflight_path: Path,
                     scene_dir: Path, factory_report: Path | None = None,
                     training_log: Path | None = None) -> dict:
    preprocessing = json.loads((preprocessing_dir / "preprocessing_report.json").read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    manifest = json.loads((scene_dir / "run_manifest.json").read_text(encoding="utf-8"))
    contract_path = scene_dir / "annotation" / "sample_annotation.json"
    heading_path = scene_dir / "annotation" / "annotation_heading_audit.json"
    surface_path = scene_dir / "annotation" / "surface_ann.json"
    object_path = scene_dir / "annotation" / "object_ann.json"
    inference_path = output_path.parent / "mask2former_inference.json"
    inference_summary = (json.loads(inference_path.read_text(encoding="utf-8"))
                         if inference_path.exists() else None)
    flow_summary_path = (output_path.parent.parent.parent / "bevlane" /
                         scene_name / "occ_flow_summary.json")
    flow_summary = (json.loads(flow_summary_path.read_text(encoding="utf-8"))
                    if flow_summary_path.exists() else None)
    heading_rows = json.loads(heading_path.read_text(encoding="utf-8")) if heading_path.exists() else []
    heading_sources: dict[str, int] = {}
    heading_classes: dict[str, dict] = {}
    for row in heading_rows:
        key = row.get("heading_source", "heading_invalid")
        heading_sources[key] = heading_sources.get(key, 0) + 1
        cls = row.get("class", "unknown")
        summary = heading_classes.setdefault(cls, {"rows": 0, "heading_valid": 0,
                                                    "heading_invalid": 0,
                                                    "source_counts": {}})
        summary["rows"] += 1
        summary["heading_valid"] += int(bool(row.get("heading_valid")))
        summary["heading_invalid"] += int(not bool(row.get("heading_valid")))
        summary["source_counts"][key] = summary["source_counts"].get(key, 0) + 1
    heading_summary = {
        "sidecar_rows": len(heading_rows),
        "heading_valid": sum(bool(row.get("heading_valid")) for row in heading_rows),
        "heading_invalid": sum(not bool(row.get("heading_valid")) for row in heading_rows),
        "source_counts": heading_sources,
        "by_class": heading_classes,
        "sidecar": str(heading_path),
    }
    heading_recomputed = manifest.get("heading_audit", {})
    car_check = heading_recomputed.get("heading_classes", {}).get("Car", {}).get("dataset_convention_check", {})
    rider_check = heading_recomputed.get("heading_classes", {}).get("Rider", {}).get("dataset_convention_check", {})
    vehicle_heading_invalid = sum(heading_classes.get(cls, {}).get("heading_invalid", 0)
                                  for cls in ("Car", "Rider"))
    pedestrian_heading_rows = heading_classes.get("Pedestrian", {}).get("rows", 0)
    training_text = (training_log.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
                     if training_log and training_log.exists() else "")
    training_summary = {
        "log": str(training_log) if training_log else None,
        "completed": "[done]" in training_text,
        "validation_has_samples": "val 0 samples" not in training_text,
        "note": "Smoke run checks one training step; no validation split was supplied."
                if "val 0 samples" in training_text else "",
    }
    anchor_camera_views = [camera for sample in preflight["samples"]
                           for camera in sample["cameras"].values()]
    anchor_freeze_views = sum(not camera.get("usable_for_crossmodal", True)
                              for camera in anchor_camera_views)
    report = {
        "scene": scene_name,
        "raw_root": manifest["raw_root"],
        "lidar_source": "LIDAR_TOP only",
        "raw_data_modified": False,
        "preprocessing": {
            "lidar_scans_seen": preprocessing["lidar_frames"],
            "lidar_scans_deskewed": preprocessing["lidar_frames_deskewed"],
            "usable_camera_sync_pairs": preprocessing["summary"]["usable_camera_sync_pairs"],
            "invalid_or_suspect_camera_pairs": preprocessing["summary"]["invalid_or_suspect_camera_pairs"],
            "dropout_intervals": preprocessing["summary"]["camera_dropout_intervals"],
            "suspected_freeze_frames": preprocessing["summary"]["suspected_freeze_frames"],
        },
        "factory_samples": preflight["accepted_count"],
        "camera_views": preflight["accepted_count"] * len(CAMERA_MAP),
        "anchor_camera_health": {
            "views": len(anchor_camera_views),
            "suspected_freeze": anchor_freeze_views,
            "usable_for_crossmodal": len(anchor_camera_views) - anchor_freeze_views,
            "note": "Freeze flags are carried as quality metadata; the montage still shows every matched view.",
        },
        "sync": preflight["sync"],
        "model_2d": "facebook/mask2former-swin-large-mapillary-vistas-panoptic",
        "mask2former_inference": inference_summary,
        "provenance_3d": "supplied PNK 3D annotations; heading/yaw recomputed from NAV-compensated tracks; not CenterPoint inference",
        "heading_recomputation": manifest.get("heading_audit"),
        "occupancy_flow": flow_summary,
        "general_annotations": {
            "surface_ann": len(json.loads(surface_path.read_text(encoding="utf-8"))),
            "object_ann": len(json.loads(object_path.read_text(encoding="utf-8"))),
            "sample_annotation": len(json.loads(contract_path.read_text(encoding="utf-8"))),
            "heading_audit": heading_summary,
        },
        "factory_report": str(factory_report) if factory_report else None,
        "training_smoke": training_summary,
        "deliverables": {
            "frames": "reports/frames/*.png",
            "video": "reports/pnk_top_lidar_top_e2e.mp4",
            "preprocessing_report": str(preprocessing_dir / "preprocessing_report.md"),
            "t4_scene": str(scene_dir),
        },
    }
    _dump(output_path.with_suffix(".json"), report)
    lines = [
        f"# LIDAR_TOP autolabel E2E report — {scene_name}", "",
        f"- Raw folder: `{manifest['raw_root']}`",
        "- LiDAR source: **LIDAR_TOP**; raw folder modified: **No**",
        f"- Deskewed scans: **{preprocessing['lidar_frames_deskewed']} / {preprocessing['lidar_frames']}**",
        f"- METEOR samples: **{preflight['accepted_count']}**; camera views: **{report['camera_views']}**",
        f"- Anchor camera views flagged freeze: **{anchor_freeze_views} / {len(anchor_camera_views)}**; "
        f"{len(anchor_camera_views) - anchor_freeze_views} marked usable for cross-modal checks.",
        f"- 2D model: **Mask2Former Swin-L Mapillary Vistas Panoptic**",
        (f"- Mask2Former views: {inference_summary.get('requested_images')} total; "
         f"{inference_summary.get('new_images')} inferred in this run, "
         f"{inference_summary.get('cache_reuse', {}).get('exact_input_cache_reused', 0)} "
         "reused from exact-image cache.") if inference_summary else "- Mask2Former inference report: unavailable.",
        "- 3D boxes: **supplied reference annotations** for flow validation; CenterPoint was not run.",
        (f"- Occupancy-flow labels: **{flow_summary['frames']} frames**; "
         f"{flow_summary['valid_agent_box_pixels']} supervised BEV cells; "
         f"{flow_summary['moving_agent_count_gt_0_5mps']} moving tracks from existing 3D annotations.")
        if flow_summary else "- Occupancy-flow labels: not generated.",
        f"- General annotations: surface={report['general_annotations']['surface_ann']}, "
        f"2D instances={report['general_annotations']['object_ann']}, "
        f"3D boxes={report['general_annotations']['sample_annotation']}",
        f"- Heading sidecar: {heading_summary['heading_valid']} resolved; "
        f"{vehicle_heading_invalid} Car/Rider boxes remain invalid; "
        f"front/back heading was not inferred for {pedestrian_heading_rows} pedestrians.",
        (f"- Heading convention check (`Heading2 + yaw`): Car median/P90 axis-motion error "
         f"{car_check.get('motion_mod180_median_deg')}°/{car_check.get('motion_mod180_p90_deg')}° "
         f"({car_check.get('moving_observations')} observations); Rider "
         f"{rider_check.get('motion_mod180_median_deg')}°/{rider_check.get('motion_mod180_p90_deg')}° "
         f"({rider_check.get('moving_observations')} observations). Pedestrian headings were not inferred."),
        f"- Training smoke completed: **{training_summary['completed']}**; "
        f"validation split available: **{training_summary['validation_has_samples']}**.",
        f"- Output frames: `{report['deliverables']['frames']}`",
        f"- MP4: `{report['deliverables']['video']}`",
        "", "## Run artifacts", "",
        f"- Preprocessing report: `{preprocessing_dir / 'preprocessing_report.md'}`",
        f"- T4 contract validation: `{output_path.parent / 't4_contract.json'}`",
        f"- Factory segmentation validation: `{factory_report or 'not available'}`",
        f"- Heading audit sidecar: `{scene_dir / 'annotation' / 'annotation_heading_audit.json'}`",
        "", "## Interpretation", "",
        "This run verifies TOP-LiDAR deskew/re-referencing → segment-local NAV motion compensation and "
        "heading resolution → Mask2Former Swin-L 2D labels → METEOR factory → occupancy-flow export "
        "→ training-smoke handoff. Supplied PNK 3D boxes remain the box source; this does not measure "
        "CenterPoint inference. The dataset-wide Heading2 + raw-yaw convention is checked against this "
        "segment, while front/back is resolved only from gated world motion and same-track propagation. "
        "Unresolved headings remain invalid in the sidecar and retain the source box rotation.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return report
