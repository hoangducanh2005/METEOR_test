"""Configuration loading for the modular lane branch."""

from __future__ import annotations

from pathlib import Path

import yaml

from .constants import CAMERAS


def load_lane_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [camera for camera in CAMERAS if camera not in config.get("cameras", {})]
    if missing:
        raise ValueError(f"Camera config is missing: {missing}")
    for camera in CAMERAS:
        item = config["cameras"][camera]
        roi = item.get("roi_normalized")
        if not isinstance(roi, list) or len(roi) != 4:
            raise ValueError(f"{camera}.roi_normalized must be [x0,y0,x1,y1]")
        if not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
            raise ValueError(f"Invalid normalized ROI for {camera}: {roi}")
        if float(item.get("reliability_weight", -1)) < 0:
            raise ValueError(f"{camera}.reliability_weight must be >= 0")
        thresholds = item.get("confidence_threshold", {})
        for detector in ("ufld_v2", "clrnet"):
            value = float(thresholds.get(detector, -1))
            if not 0 <= value <= 1:
                raise ValueError(f"Invalid {detector} threshold for {camera}: {value}")
    return config

