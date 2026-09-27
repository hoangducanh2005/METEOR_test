"""Read the PNK intrinsic/extrinsic calibration bundle."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np


def load_calibration(calibration_dir: Path, vehicle: str, recording_date: date) -> tuple[dict, dict, str]:
    intr_path = calibration_dir / f"{vehicle}_Intrinsics.json"
    extr_path = calibration_dir / f"{vehicle}_Extrinsics_By_Dates.json"
    intrinsics = json.loads(intr_path.read_text(encoding="utf-8-sig"))
    periods = json.loads(extr_path.read_text(encoding="utf-8-sig"))["periods"]
    chosen = None
    for period in periods:
        start = date.fromisoformat(period["valid_from"])
        end = date.fromisoformat(period["valid_to"])
        if start <= recording_date < end:
            chosen = period
            break
    if chosen is None:
        raise ValueError(f"No extrinsic period covers {recording_date.isoformat()}")
    return intrinsics, chosen["extrinsics"], chosen["id"]


def camera_to_ego(extrinsic_ego_to_camera: list[list[float]]) -> np.ndarray:
    """PNK camera matrices map ego -> camera; T4 stores camera -> ego."""
    return np.linalg.inv(np.asarray(extrinsic_ego_to_camera, dtype=np.float64))

