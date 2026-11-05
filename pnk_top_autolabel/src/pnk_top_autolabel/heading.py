"""Run-local world-frame heading audit for PNK reference boxes.

The selected dataset convention (NAV Heading2 + raw box yaw) comes from the
previous full-dataset audit. This module recomputes object motion in ENU for
the current segment, checks all yaw/NAV candidates as diagnostics, and resolves
the 180-degree front/back branch only from gated motion or same-track evidence.
Raw labels are read-only.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DATASET_NAV_FIELD = "Heading2"
DATASET_YAW_CANDIDATE = "yaw"
HEADING_CLASSES = {"Car", "Rider"}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _wa():
    """Import the established audit helpers without making them a CLI dependency."""
    repo_root = Path(__file__).resolve().parents[3]
    audit_dir = repo_root / "dataset_check"
    if str(audit_dir) not in sys.path:
        sys.path.insert(0, str(audit_dir))
    import world_yaw_audit
    return world_yaw_audit


def build_session_heading_audit(raw_root: Path, label_dir: Path,
                                output_dir: Path) -> dict:
    """Create a local heading CSV and candidate report for one labeled segment."""
    wa = _wa()
    raw_root, label_dir = raw_root.resolve(), label_dir.resolve()
    labels = sorted(label_dir.glob("*.txt"), key=lambda p: wa.parse_ts(p.stem))
    if not labels:
        raise FileNotFoundError(f"No reference label files in {label_dir}")
    timestamps = [wa.parse_ts(path.stem) for path in labels]
    nav = wa.load_nav(raw_root, timestamps)
    groups_by_field = {field: defaultdict(list) for field in wa.NAV_FIELDS}
    groups_by_class_field = {
        cls: {field: defaultdict(list) for field in wa.NAV_FIELDS}
        for cls in HEADING_CLASSES
    }
    class_boxes: Counter[str] = Counter()
    malformed = 0
    nav_matched = 0
    nav_rejected = 0

    for path, timestamp in zip(labels, timestamps):
        pose = nav.interpolate(timestamp)
        if pose is None:
            nav_rejected += 1
            continue
        nav_matched += 1
        pose["enu"] = nav.to_enu(pose["lat"], pose["lon"], pose["alt"])
        _, boxes, bad = wa.base.read_label_file(path)
        malformed += bad
        for box in boxes:
            if box.cls not in HEADING_CLASSES:
                continue
            class_boxes[box.cls] += 1
            common = {
                "segment": label_dir.parent.name,
                "timestamp": box.timestamp,
                "timestamp_s": box.timestamp_s,
                "trackid": box.trackid,
                "class": box.cls,
                "x": box.x,
                "y": box.y,
                "z": box.z,
                "yaw": box.yaw,
                "numpoints": box.numpoints,
            }
            # Class is included in the internal key so a reused numeric track ID
            # across categories cannot join unrelated trajectories.
            key = (common["segment"], box.cls, box.trackid)
            for field in wa.NAV_FIELDS:
                obs = wa.add_world_pose(common, nav, pose, field)
                if obs is None:
                    continue
                groups_by_field[field][key].append(obs)
                groups_by_class_field[box.cls][field][key].append(obs)

    for groups in groups_by_field.values():
        for track in groups.values():
            wa.estimate_motion(track)
    for by_field in groups_by_class_field.values():
        for groups in by_field.values():
            for track in groups.values():
                wa.estimate_motion(track)

    # Keep the dataset-wide convention established by the earlier audit; this
    # segment's candidate tables are evidence/checks, not an unstable per-scene
    # convention switch (especially important for the Rider-heavy sample).
    config = {
        "nav_heading_field": DATASET_NAV_FIELD,
        "yaw_candidate": DATASET_YAW_CANDIDATE,
    }
    rows = wa.candidate_table(groups_by_field)
    selected = next((row for row in rows
                     if row["nav_heading_field"] == DATASET_NAV_FIELD
                     and row["yaw_candidate"] == DATASET_YAW_CANDIDATE), {})
    class_summaries = {}
    for cls, by_field in groups_by_class_field.items():
        candidates = wa.candidate_table(by_field)
        ranked = [row for row in candidates if row.get("moving_observations", 0) > 0]
        best = min(ranked, key=lambda row: (
            row.get("motion_mod180_median_deg", 999),
            row.get("static_axis_jitter_track_median_deg", 999),
            -row.get("moving_observations", 0),
        )) if ranked else {}
        fixed = next((row for row in candidates
                      if row["nav_heading_field"] == DATASET_NAV_FIELD
                      and row["yaw_candidate"] == DATASET_YAW_CANDIDATE), {})
        class_summaries[cls] = {
            "boxes": class_boxes[cls],
            "tracks": len({key[-1] for key in by_field[DATASET_NAV_FIELD]}),
            "best_local_candidate": {
                "nav_heading_field": best.get("nav_heading_field"),
                "yaw_candidate": best.get("yaw_candidate"),
                "moving_observations": best.get("moving_observations", 0),
                "motion_mod180_median_deg": best.get("motion_mod180_median_deg"),
                "motion_mod180_p90_deg": best.get("motion_mod180_p90_deg"),
            },
            "dataset_convention_check": {
                "nav_heading_field": DATASET_NAV_FIELD,
                "yaw_candidate": DATASET_YAW_CANDIDATE,
                "moving_observations": fixed.get("moving_observations", 0),
                "motion_mod180_median_deg": fixed.get("motion_mod180_median_deg"),
                "motion_mod180_p90_deg": fixed.get("motion_mod180_p90_deg"),
            },
        }

    resolved = wa.resolve_track_headings(groups_by_field[DATASET_NAV_FIELD], config)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_csv = output_dir / "heading_candidate_comparison.csv"
    resolved_csv = output_dir / "resolved_world_headings.csv"
    wa.write_csv(candidate_csv, rows)
    wa.write_csv(resolved_csv, resolved)
    source_counts = Counter(row["heading_source"] for row in resolved)
    summary = {
        "raw_root": str(raw_root),
        "label_dir": str(label_dir),
        "segment": label_dir.parent.name,
        "raw_data_modified": False,
        "label_frames": len(labels),
        "nav_frames_matched": nav_matched,
        "nav_frames_rejected": nav_rejected,
        "malformed_label_lines": malformed,
        "heading_classes": class_summaries,
        "dataset_convention": config,
        "combined_dataset_convention_check": {
            "moving_observations": selected.get("moving_observations", 0),
            "motion_mod180_median_deg": selected.get("motion_mod180_median_deg"),
            "motion_mod180_p90_deg": selected.get("motion_mod180_p90_deg"),
            "motion_mod180_within_25deg_pct": selected.get("motion_mod180_lt25_pct"),
        },
        "resolved_rows": len(resolved),
        "heading_valid": sum(bool(row["heading_valid"]) for row in resolved),
        "heading_invalid": sum(not bool(row["heading_valid"]) for row in resolved),
        "heading_source_counts": dict(source_counts),
        "candidate_csv": str(candidate_csv),
        "resolved_csv": str(resolved_csv),
        "method": "NAV ego-motion compensated to local ENU; moving objects choose yaw or yaw+180; same-track propagation limited to contiguous gaps <=0.75s; unresolved remains heading_invalid.",
    }
    _write_json(output_dir / "heading_audit_summary.json", summary)
    return summary
