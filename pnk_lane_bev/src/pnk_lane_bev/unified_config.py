"""Configuration and taxonomy resolution for the unified segmentation pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml

from .constants import CAMERAS


def load_unified_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported config schema in {path}")
    for group in ("drivable_classes", "lane_classes"):
        values = config.get("taxonomy", {}).get(group)
        if not isinstance(values, list) or not values:
            raise ValueError(f"taxonomy.{group} must be a non-empty list")
    missing = [camera for camera in CAMERAS if camera not in config.get("cameras", {})]
    if missing:
        raise ValueError(f"Unified segmentation config is missing cameras: {missing}")
    for camera in CAMERAS:
        item = config["cameras"][camera]
        weight = float(item.get("reliability_weight", -1))
        threshold = float(item.get("lane_confidence_threshold", -1))
        if weight < 0:
            raise ValueError(f"{camera}.reliability_weight must be >= 0")
        if not 0 <= threshold <= 1:
            raise ValueError(f"{camera}.lane_confidence_threshold must be in [0,1]")
    for name in ("drivable_threshold", "lane_threshold", "semantic_confidence_floor"):
        value = float(config["fusion"][name])
        if not 0 <= value <= 1:
            raise ValueError(f"fusion.{name} must be in [0,1]")
    return config


def resolve_class_ids(id2label: dict, requested: list[str], group: str) -> dict[int, str]:
    labels = {int(key): str(value) for key, value in id2label.items()}
    by_name = {value.strip().casefold(): (key, value) for key, value in labels.items()}
    missing = [name for name in requested if name.strip().casefold() not in by_name]
    if missing:
        raise ValueError(f"Checkpoint taxonomy does not contain {group}: {missing}")
    return {by_name[name.strip().casefold()][0]: by_name[name.strip().casefold()][1] for name in requested}
