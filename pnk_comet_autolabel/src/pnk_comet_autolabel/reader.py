"""PNKData discovery, timestamp matching and ego-pose interpolation."""

from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left
from pathlib import Path

import numpy as np

from .constants import CAMERA_MAP


def timestamp_ns(path_or_name: str | Path) -> int:
    stem = Path(path_or_name).stem
    sec, nsec = stem.split("-", 1)
    return int(sec) * 1_000_000_000 + int(nsec.ljust(9, "0")[:9])


def nearest_path(entries: list[tuple[int, Path]], target: int) -> tuple[int, Path]:
    stamps = [item[0] for item in entries]
    at = bisect_left(stamps, target)
    candidates = entries[max(0, at - 1): min(len(entries), at + 1)]
    if not candidates:
        raise ValueError("empty timestamp stream")
    return min(candidates, key=lambda item: abs(item[0] - target))


def scan_pnk(raw_root: Path, max_delta_ms: float = 35.0) -> dict:
    raw_root = raw_root.resolve()
    lidar_dir = raw_root / "LIDAR" / "LIDAR_MERGED_EGO"
    camera_root = raw_root / "CAMERA"
    lidar = sorted((timestamp_ns(p), p) for p in lidar_dir.glob("*.laz"))
    if not lidar:
        raise FileNotFoundError(f"No merged LiDAR .laz files under {lidar_dir}")
    cameras: dict[str, list[tuple[int, Path]]] = {}
    for source in CAMERA_MAP:
        entries = sorted((timestamp_ns(p), p) for p in (camera_root / source).glob("*.jpg"))
        if not entries:
            raise FileNotFoundError(f"No camera images under {camera_root / source}")
        cameras[source] = entries

    used: dict[str, set[str]] = {name: set() for name in CAMERA_MAP}
    samples = []
    dropped = []
    deltas_ms = []
    for stamp, lidar_path in lidar:
        matched = {}
        reason = None
        for source, target in CAMERA_MAP.items():
            image_stamp, image_path = nearest_path(cameras[source], stamp)
            delta_ms = abs(image_stamp - stamp) / 1e6
            if delta_ms > max_delta_ms:
                reason = f"{source} delta {delta_ms:.3f} ms > {max_delta_ms:.3f} ms"
            if str(image_path) in used[source]:
                reason = f"{source} image reused: {image_path.name}"
            matched[target] = {
                "source_channel": source,
                "source_path": str(image_path),
                "timestamp_ns": image_stamp,
                "delta_ms": delta_ms,
            }
            deltas_ms.append(delta_ms)
        if reason:
            dropped.append({"timestamp_ns": stamp, "reason": reason})
            continue
        for target, info in matched.items():
            used[info["source_channel"]].add(info["source_path"])
        samples.append({
            "timestamp_ns": stamp,
            "lidar_path": str(lidar_path),
            "cameras": matched,
        })

    spreads = [max(v["timestamp_ns"] for v in s["cameras"].values())
               - min(v["timestamp_ns"] for v in s["cameras"].values()) for s in samples]
    return {
        "raw_root": str(raw_root),
        "lidar_input_count": len(lidar),
        "camera_input_counts": {k: len(v) for k, v in cameras.items()},
        "accepted_count": len(samples),
        "dropped_count": len(dropped),
        "dropped": dropped,
        "sync": {
            "max_delta_ms": max(deltas_ms) if deltas_ms else None,
            "p95_delta_ms": float(np.percentile(deltas_ms, 95)) if deltas_ms else None,
            "max_intercamera_spread_ms": max(spreads) / 1e6 if spreads else None,
            "threshold_ms": max_delta_ms,
        },
        "samples": samples,
    }


def write_preflight(raw_root: Path, output: Path, max_delta_ms: float = 35.0) -> dict:
    result = scan_pnk(raw_root, max_delta_ms=max_delta_ms)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


class PoseInterpolator:
    """Local ENU pose sampled from NAV CSV rows inside the scene window."""

    def __init__(self, raw_root: Path, t_min_ns: int, t_max_ns: int, margin_s: float = 2.0):
        lo, hi = t_min_ns - int(margin_s * 1e9), t_max_ns + int(margin_s * 1e9)
        rows = []
        total = 0
        for path in sorted((raw_root / "OTHERS" / "NAV").glob("*.csv")):
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    total += 1
                    stamp = timestamp_ns(row["Timestamp"])
                    if lo <= stamp <= hi:
                        rows.append((stamp, float(row["Latitude"]), float(row["Longitude"]),
                                     float(row["Altitude"]), float(row["Heading2"])))
        if len(rows) < 2:
            raise ValueError("NAV coverage is missing for the requested scene window")
        rows.sort(key=lambda item: item[0])
        arr = np.asarray(rows, dtype=np.float64)
        self.timestamps = arr[:, 0]
        lat, lon, altitude = arr[:, 1], arr[:, 2], arr[:, 3]
        lat0, lon0, alt0 = lat[0], lon[0], altitude[0]
        radius = 6_378_137.0
        self.east = np.deg2rad(lon - lon0) * radius * math.cos(math.radians(lat0))
        self.north = np.deg2rad(lat - lat0) * radius
        self.up = altitude - alt0
        # Heading is clockwise from north; T4 yaw is counter-clockwise from east.
        self.yaw = np.unwrap(np.deg2rad(90.0 - arr[:, 4]))
        self.stats = {
            "rows_scanned": total,
            "rows_kept": len(rows),
            "coverage_ns": [int(self.timestamps[0]), int(self.timestamps[-1])],
            "origin_wgs84": [float(lat0), float(lon0), float(alt0)],
        }

    def at(self, stamp_ns: int) -> tuple[list[float], float]:
        if stamp_ns < self.timestamps[0] or stamp_ns > self.timestamps[-1]:
            raise ValueError(f"Pose timestamp {stamp_ns} is outside NAV coverage")
        xyz = [float(np.interp(stamp_ns, self.timestamps, v))
               for v in (self.east, self.north, self.up)]
        yaw = float(np.interp(stamp_ns, self.timestamps, self.yaw))
        return xyz, yaw

