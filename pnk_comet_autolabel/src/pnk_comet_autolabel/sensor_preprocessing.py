"""Read-only LIDAR_TOP preprocessing, camera sync audit and motion deskew.

The raw PNK session is never modified. All point clouds, manifests, reports and
overlays are written to a required output directory outside the raw session.

Motion model: NAV position plus Heading2/Pitch/Roll are interpolated at every
point timestamp. Each LIDAR_TOP point is transformed through local ENU and
deskewed to the scan-end ego pose. For camera overlays, that point is then
expressed in the ego pose at the selected camera timestamp and projected with
the camera's native distortion model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import cv2
import laspy
import numpy as np

from .calibration import load_calibration
from .constants import CAMERA_MAP
from .reader import timestamp_ns


WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
DEFAULT_GOOD_SYNC_MS = 15.0
DEFAULT_MAX_SYNC_MS = 35.0
DEFAULT_DROPOUT_MS = 50.0
DEFAULT_MAX_NAV_GAP_MS = 50.0


@dataclass
class PoseBatch:
    position_enu: np.ndarray
    rotation_world_ego: np.ndarray
    heading_std_deg: np.ndarray
    bracket_gap_ms: np.ndarray
    valid: np.ndarray


def _parse_session_date(raw_root: Path, first_stamp_ns: int) -> date:
    match = re.match(r"^(\d{4})(\d{2})(\d{2})_", raw_root.name)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return datetime.fromtimestamp(first_stamp_ns / 1e9, tz=timezone.utc).date()


def _read_nav(raw_root: Path, low_ns: int, high_ns: int) -> dict[str, np.ndarray]:
    rows = []
    nav_dir = raw_root / "OTHERS" / "NAV"
    if not nav_dir.is_dir():
        raise FileNotFoundError(f"NAV folder not found: {nav_dir}")
    for path in sorted(nav_dir.glob("*.csv")):
        # NAV files are about one-minute chunks; skip files outside the window.
        try:
            chunk_start = timestamp_ns(path.stem)
        except (ValueError, TypeError):
            chunk_start = None
        if chunk_start is not None and (chunk_start > high_ns + 2_000_000_000
                                        or chunk_start + 62_000_000_000 < low_ns - 2_000_000_000):
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    stamp = timestamp_ns(row["Timestamp"])
                    if stamp < low_ns - 2_000_000_000 or stamp > high_ns + 2_000_000_000:
                        continue
                    values = [float(row[k]) for k in
                              ("Latitude", "Longitude", "Altitude", "Heading2", "Pitch", "Roll")]
                    if not all(math.isfinite(v) for v in values):
                        continue
                    std = float(row.get("Heading2_std", "nan"))
                    rows.append((stamp, *values, std))
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
    rows.sort(key=lambda r: r[0])
    unique = []
    for row in rows:
        if not unique or row[0] > unique[-1][0]:
            unique.append(row)
    if len(unique) < 2:
        raise RuntimeError(f"Insufficient NAV pose coverage near {low_ns}..{high_ns}")
    arr = np.asarray(unique, dtype=np.float64)
    return {
        "timestamp_ns": arr[:, 0].astype(np.int64),
        "latitude": arr[:, 1], "longitude": arr[:, 2], "altitude": arr[:, 3],
        "heading2_deg": arr[:, 4], "pitch_deg": arr[:, 5], "roll_deg": arr[:, 6],
        "heading2_std_deg": arr[:, 7],
    }


def _lla_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray,
                 altitude_m: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + altitude_m) * np.cos(lat) * np.cos(lon)
    y = (n + altitude_m) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + altitude_m) * np.sin(lat)
    return np.column_stack((x, y, z))


def _rotation_world_ego(yaw: np.ndarray, pitch: np.ndarray,
                        roll: np.ndarray) -> np.ndarray:
    """Rz(yaw) Ry(-pitch) Rx(roll), ego axes forward-left-up, world ENU."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    n = len(yaw)
    rz = np.zeros((n, 3, 3), dtype=np.float64)
    ry = np.zeros_like(rz)
    rx = np.zeros_like(rz)
    rz[:, 0, 0], rz[:, 0, 1] = cy, -sy
    rz[:, 1, 0], rz[:, 1, 1] = sy, cy
    rz[:, 2, 2] = 1.0
    # Positive NAV pitch is nose-up; with forward-left-up axes that is -Y.
    ry[:, 0, 0], ry[:, 0, 2] = cp, -sp
    ry[:, 1, 1] = 1.0
    ry[:, 2, 0], ry[:, 2, 2] = sp, cp
    rx[:, 0, 0] = 1.0
    rx[:, 1, 1], rx[:, 1, 2] = cr, -sr
    rx[:, 2, 1], rx[:, 2, 2] = sr, cr
    return np.einsum("nij,njk,nkl->nil", rz, ry, rx, optimize=True)


class NavTimeline:
    def __init__(self, rows: dict[str, np.ndarray], max_gap_ms: float):
        self.t_ns = rows["timestamp_ns"]
        self.t_s = (self.t_ns - self.t_ns[0]).astype(np.float64) / 1e9
        lat, lon, alt = rows["latitude"], rows["longitude"], rows["altitude"]
        self.origin_lat = float(lat[0])
        self.origin_lon = float(lon[0])
        ecef = _lla_to_ecef(lat, lon, alt)
        lat0, lon0 = math.radians(self.origin_lat), math.radians(self.origin_lon)
        delta = ecef - ecef[0]
        east = -math.sin(lon0) * delta[:, 0] + math.cos(lon0) * delta[:, 1]
        north = (-math.sin(lat0) * math.cos(lon0) * delta[:, 0]
                 - math.sin(lat0) * math.sin(lon0) * delta[:, 1]
                 + math.cos(lat0) * delta[:, 2])
        up = (math.cos(lat0) * math.cos(lon0) * delta[:, 0]
              + math.cos(lat0) * math.sin(lon0) * delta[:, 1]
              + math.sin(lat0) * delta[:, 2])
        self.position = np.column_stack((east, north, up))
        # Compass degrees clockwise from North -> ENU yaw CCW from East.
        self.yaw = np.unwrap(np.deg2rad(90.0 - rows["heading2_deg"]))
        self.pitch = np.deg2rad(rows["pitch_deg"])
        self.roll = np.deg2rad(rows["roll_deg"])
        self.heading_std = rows["heading2_std_deg"]
        self.max_gap_ms = max_gap_ms

    def at(self, query_ns: np.ndarray | int) -> PoseBatch:
        q_ns = np.atleast_1d(np.asarray(query_ns, dtype=np.int64))
        q_s = (q_ns - self.t_ns[0]).astype(np.float64) / 1e9
        hi = np.searchsorted(self.t_s, q_s, side="left")
        hi_clip = np.clip(hi, 1, len(self.t_s) - 1)
        lo_clip = hi_clip - 1
        exact0 = hi == 0
        exact_last = hi == len(self.t_s)
        # Exact samples use their actual timestamp as both bracket endpoints.
        exact = np.zeros(len(q_s), dtype=bool)
        in_range = (hi < len(self.t_s))
        exact[in_range] = np.abs(self.t_s[hi[in_range]] - q_s[in_range]) < 1e-9
        lo = np.where(exact, hi, lo_clip)
        upper = np.where(exact, hi, hi_clip)
        gap_s = self.t_s[upper] - self.t_s[lo]
        gap_ms = gap_s * 1000.0
        valid = (~exact0 & ~exact_last) | exact
        valid &= gap_ms <= self.max_gap_ms
        denom = np.where(gap_s > 0, gap_s, 1.0)
        w = np.clip((q_s - self.t_s[lo]) / denom, 0.0, 1.0)

        def lerp(values: np.ndarray) -> np.ndarray:
            return values[lo] + w * (values[upper] - values[lo])

        pos = np.column_stack([lerp(self.position[:, k]) for k in range(3)])
        yaw = lerp(self.yaw)
        pitch, roll = lerp(self.pitch), lerp(self.roll)
        std = lerp(np.nan_to_num(self.heading_std, nan=999.0))
        rotation = _rotation_world_ego(yaw, pitch, roll)
        return PoseBatch(pos, rotation, std, gap_ms, valid)


def _point_timestamps_ns(raw_values: np.ndarray, frame_ns: int,
                         mode: str = "auto") -> tuple[np.ndarray, str]:
    values = np.asarray(raw_values)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("point_timestamp must be numeric")
    as_float = values.astype(np.float64)
    if mode in ("auto", "epoch_ns") and np.nanmedian(as_float) > 1e17:
        stamps = values.astype(np.int64)
        offsets = stamps - np.int64(frame_ns)
        if np.max(np.abs(offsets)) <= 500_000_000:
            return stamps, "epoch_ns"
        if mode == "epoch_ns":
            raise ValueError("point_timestamp epoch-ns values are not near the filename timestamp")
    if mode == "auto":
        max_abs = float(np.nanmax(np.abs(as_float)))
        if max_abs <= 1.0:
            mode = "offset_s"
        elif max_abs <= 1e9:
            mode = "offset_ns"
        else:
            raise ValueError("Cannot infer point_timestamp unit; set --point-time-mode")
    if mode == "epoch_ns":
        stamps = values.astype(np.int64)
        if np.max(np.abs(stamps - np.int64(frame_ns))) > 500_000_000:
            raise ValueError("point_timestamp epoch-ns values are not near the filename timestamp")
        return stamps, "epoch_ns"
    scale = 1e9 if mode == "offset_s" else 1.0
    offset_ns = np.rint(as_float * scale).astype(np.int64)
    # Offset arrays typically start at zero and end at scan duration; filename
    # timestamp is the scan-end time, so align the maximum offset to that time.
    stamps = np.int64(frame_ns) - int(np.max(offset_ns)) + offset_ns
    return stamps, mode


def _camera_files(raw_root: Path) -> dict[str, list[tuple[int, Path]]]:
    result = {}
    root = raw_root / "CAMERA"
    for channel in CAMERA_MAP:
        folder = root / channel
        if not folder.is_dir():
            result[channel] = []
            continue
        files = []
        for path in folder.iterdir():
            if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                try:
                    files.append((timestamp_ns(path), path))
                except ValueError:
                    continue
        files.sort(key=lambda item: item[0])
        result[channel] = files
    return result


def _average_hash(image_path: Path) -> int | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    small = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    bits = (small >= mean).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _image_health(camera_files: dict[str, list[tuple[int, Path]]],
                  nav: NavTimeline, dropout_ms: float) -> tuple[dict, list[dict], list[dict], list[dict]]:
    freeze_by_stamp: dict[str, set[int]] = {}
    quality_rows, dropout_rows, freeze_rows = [], [], []
    for channel, files in camera_files.items():
        if not files:
            quality_rows.append({"channel": channel, "image_count": 0,
                                 "median_period_ms": None, "p95_period_ms": None,
                                 "max_gap_ms": None, "dropout_gaps_over_threshold": 0,
                                 "suspected_freeze_frames": 0, "status": "missing_channel"})
            freeze_by_stamp[channel] = set()
            continue
        times = np.asarray([x[0] for x in files], dtype=np.int64)
        hashes = [_average_hash(path) for _, path in files]
        gaps_ms = np.diff(times).astype(np.float64) / 1e6
        poses = nav.at(times)
        moved = poses.position_enu
        moved_yaw = np.arctan2(poses.rotation_world_ego[:, 1, 0],
                               poses.rotation_world_ego[:, 0, 0])
        freeze_indices = set()
        for i, gap in enumerate(gaps_ms):
            if gap > dropout_ms:
                dropout_rows.append({"channel": channel, "start_timestamp": files[i][1].stem,
                                     "end_timestamp": files[i + 1][1].stem,
                                     "gap_ms": float(gap)})
            if hashes[i] is None or hashes[i + 1] is None:
                continue
            if not (poses.valid[i] and poses.valid[i + 1]):
                continue
            displacement = float(np.linalg.norm(moved[i + 1, :2] - moved[i, :2]))
            yaw_change = abs((moved_yaw[i + 1] - moved_yaw[i] + math.pi) % (2 * math.pi) - math.pi)
            hash_distance = _hamming(hashes[i], hashes[i + 1])
            if (hash_distance == 0 and (displacement > 0.10 or yaw_change > math.radians(0.5))) or (
                    hash_distance <= 1 and (displacement > 0.30 or yaw_change > math.radians(1.0))):
                freeze_indices.add(i + 1)
                freeze_rows.append({"channel": channel, "timestamp": files[i + 1][1].stem,
                                    "previous_timestamp": files[i][1].stem,
                                    "ego_translation_m": displacement,
                                    "ego_yaw_delta_deg": math.degrees(yaw_change),
                                    "hash_distance": hash_distance,
                                    "classification": "suspected_freeze"})
        freeze_by_stamp[channel] = freeze_indices
        quality_rows.append({"channel": channel, "image_count": len(files),
                             "median_period_ms": float(np.median(gaps_ms)) if len(gaps_ms) else None,
                             "p95_period_ms": float(np.percentile(gaps_ms, 95)) if len(gaps_ms) else None,
                             "max_gap_ms": float(np.max(gaps_ms)) if len(gaps_ms) else None,
                             "dropout_gaps_over_threshold": int(np.sum(gaps_ms > dropout_ms)),
                             "suspected_freeze_frames": len(freeze_indices)})
    return freeze_by_stamp, quality_rows, dropout_rows + [], freeze_rows


def _select_camera(files: list[tuple[int, Path]], target_ns: int) -> tuple[int, Path, int]:
    stamps = [item[0] for item in files]
    idx = int(np.searchsorted(stamps, target_ns, side="left"))
    choices = files[max(0, idx - 1):min(len(files), idx + 1)]
    if not choices:
        raise RuntimeError("Camera file list is empty")
    stamp, path = min(choices, key=lambda item: abs(item[0] - target_ns))
    return stamp, path, stamp - target_ns


def _matrix(value, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    det = float(np.linalg.det(matrix[:3, :3]))
    if not 0.95 <= det <= 1.05:
        raise ValueError(f"{name} rotation determinant is {det:.4f}; transform convention/data invalid")
    return matrix


def _overlay(image_path: Path, points_camera: np.ndarray, intrinsic: np.ndarray,
             distortion: np.ndarray, target_dir: Path, stem: str, channel: str,
             expected_size: tuple[int, int]) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read camera image {image_path}")
    expected_w, expected_h = expected_size
    actual_h, actual_w = image.shape[:2]
    if (actual_w, actual_h) != (expected_w, expected_h):
        expected_aspect = expected_w / expected_h
        actual_aspect = actual_w / actual_h
        if abs(expected_aspect - actual_aspect) > 1e-3:
            raise ValueError(f"Image shape {actual_w}x{actual_h} disagrees with calibration "
                             f"{expected_w}x{expected_h}; aspect-changing crop/resize is unknown")
        scale = np.diag([actual_w / expected_w, actual_h / expected_h, 1.0])
        intrinsic = scale @ intrinsic
    z = points_camera[:, 2]
    keep = (z > 0.1) & (z < 120.0)
    points = points_camera[keep]
    z = z[keep]
    if not len(points):
        return
    projected, _ = cv2.projectPoints(points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                     intrinsic, distortion)
    uv = projected.reshape(-1, 2)
    height, width = image.shape[:2]
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < width)
              & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    uv, z = uv[inside], z[inside]
    if len(uv) > 30000:
        chosen = np.linspace(0, len(uv) - 1, 30000, dtype=np.int64)
        uv, z = uv[chosen], z[chosen]
    colors = cv2.applyColorMap(np.clip(255.0 * (z - 2.0) / 80.0, 0, 255).astype(np.uint8),
                               cv2.COLORMAP_TURBO).reshape(-1, 3)
    raw_overlay = image.copy()
    for (u, v), color in zip(uv.astype(np.int32), colors):
        cv2.circle(raw_overlay, (int(u), int(v)), 1, tuple(int(c) for c in color), -1)

    undistorted = cv2.undistort(image, intrinsic, distortion, None, intrinsic)
    uv_undist, _ = cv2.projectPoints(points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                     intrinsic, np.zeros_like(distortion))
    uv_undist = uv_undist.reshape(-1, 2)
    inside_u = ((uv_undist[:, 0] >= 0) & (uv_undist[:, 0] < width)
                & (uv_undist[:, 1] >= 0) & (uv_undist[:, 1] < height))
    rect_overlay = undistorted.copy()
    uv2 = uv_undist[inside_u]
    z2 = points[:, 2][inside_u]
    if len(uv2) > 30000:
        chosen = np.linspace(0, len(uv2) - 1, 30000, dtype=np.int64)
        uv2, z2 = uv2[chosen], z2[chosen]
    colors2 = cv2.applyColorMap(np.clip(255.0 * (z2 - 2.0) / 80.0, 0, 255).astype(np.uint8),
                                cv2.COLORMAP_TURBO).reshape(-1, 3)
    for (u, v), color in zip(uv2.astype(np.int32), colors2):
        cv2.circle(rect_overlay, (int(u), int(v)), 1, tuple(int(c) for c in color), -1)
    canvas = np.concatenate((raw_overlay, rect_overlay), axis=1)
    cv2.putText(canvas, f"RAW / distorted projection: {channel}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "UNDISTORTED / zero-distortion projection", (width + 20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    target_dir.mkdir(parents=True, exist_ok=True)
    output = target_dir / f"{stem}_{channel}.jpg"
    if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not write {output}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def preprocess_session(raw_root: Path, calibration_dir: Path, output_dir: Path,
                       point_time_mode: str = "auto", limit: int | None = None,
                       visualization_stride: int = 10, good_sync_ms: float = DEFAULT_GOOD_SYNC_MS,
                       max_sync_ms: float = DEFAULT_MAX_SYNC_MS,
                       dropout_ms: float = DEFAULT_DROPOUT_MS,
                       max_nav_gap_ms: float = DEFAULT_MAX_NAV_GAP_MS) -> dict:
    raw_root = raw_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir == raw_root or raw_root in output_dir.parents or output_dir in raw_root.parents:
        raise ValueError("Output must be outside the raw dataset folder and its parent folders")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output folder is not empty; choose a new staging folder: {output_dir}")
    top_dir = raw_root / "LIDAR" / "LIDAR_TOP"
    lidar_files = sorted(((timestamp_ns(p), p) for p in top_dir.glob("*.laz")), key=lambda x: x[0])
    if not lidar_files:
        raise FileNotFoundError(f"No LIDAR_TOP .laz files under {top_dir}")
    if limit is not None:
        lidar_files = lidar_files[:limit]

    camera_files = _camera_files(raw_root)
    all_camera_stamps = [stamp for files in camera_files.values() for stamp, _ in files]
    if not all_camera_stamps:
        all_camera_stamps = [stamp for stamp, _ in lidar_files]
    first_ns, last_ns = lidar_files[0][0], lidar_files[-1][0]
    nav_rows = _read_nav(raw_root, min(first_ns, min(all_camera_stamps)),
                         max(last_ns, max(all_camera_stamps)))
    nav = NavTimeline(nav_rows, max_nav_gap_ms=max_nav_gap_ms)
    session_date = _parse_session_date(raw_root, first_ns)
    intrinsics, extrinsics, period_id = load_calibration(calibration_dir, "VF6_01", session_date)
    top_to_ego = _matrix(extrinsics["LIDAR_TOP"], "T_EGO_FROM_LIDAR_TOP")
    camera_cal = {}
    for channel in CAMERA_MAP:
        if channel not in intrinsics or channel not in extrinsics:
            raise KeyError(f"Calib_2 has no calibration for camera {channel}")
        # Existing PNK camera matrices map ego -> camera. LIDAR_TOP is documented
        # as sensor -> ego. Keep these directions explicit; never invert both.
        camera_cal[channel] = {
            "ego_to_camera": _matrix(extrinsics[channel], f"T_{channel}_FROM_EGO"),
            "K": np.asarray(intrinsics[channel]["camera_matrix"], dtype=np.float64),
            "dist": np.asarray(intrinsics[channel]["distortion_coefficients"], dtype=np.float64).reshape(-1, 1),
            "expected_size": (int(intrinsics[channel]["image_width"]),
                              int(intrinsics[channel]["image_height"])),
        }

    freeze_by_stamp, camera_quality, dropout_rows, freeze_rows = _image_health(
        camera_files, nav, dropout_ms)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = output_dir / "lidar_top_deskewed_ego"
    processed_dir.mkdir(parents=True, exist_ok=True)
    sync_rows, frame_rows = [], []
    used_images = {channel: set() for channel in CAMERA_MAP}
    success_count = 0

    for index, (frame_ns, lidar_path) in enumerate(lidar_files):
        frame_stamp = lidar_path.stem
        frame_status = "ok"
        try:
            cloud = laspy.read(lidar_path)
            dims = set(cloud.point_format.dimension_names)
            if "point_timestamp" not in dims:
                raise ValueError("LIDAR_TOP has no point_timestamp dimension")
            point_ns, time_mode = _point_timestamps_ns(
                np.asarray(cloud["point_timestamp"]), frame_ns, point_time_mode)
            point_pose = nav.at(point_ns)
            end_pose = nav.at(frame_ns)
            if not np.all(point_pose.valid) or not bool(end_pose.valid[0]):
                bad = int(np.sum(~point_pose.valid))
                raise ValueError(f"NAV interpolation gap/coverage failure ({bad} invalid point poses)")
            if np.any(point_pose.heading_std_deg > 2.0) or end_pose.heading_std_deg[0] > 2.0:
                raise ValueError("NAV Heading2_std exceeds 2 degrees")

            p_top = np.column_stack((np.asarray(cloud.x, dtype=np.float64),
                                     np.asarray(cloud.y, dtype=np.float64),
                                     np.asarray(cloud.z, dtype=np.float64)))
            r_top, t_top = top_to_ego[:3, :3], top_to_ego[:3, 3]
            p_ego_i = p_top @ r_top.T + t_top
            p_world = np.einsum("nij,nj->ni", point_pose.rotation_world_ego, p_ego_i,
                                optimize=True) + point_pose.position_enu
            r_end = end_pose.rotation_world_ego[0]
            p_ego_end = (p_world - end_pose.position_enu[0]) @ r_end
            intensity = np.asarray(cloud.intensity, dtype=np.float32)
            np.savez_compressed(processed_dir / f"{frame_stamp}.npz",
                                xyz_ego_at_scan_end=p_ego_end.astype(np.float32),
                                intensity=intensity, point_timestamp_ns=point_ns,
                                scan_end_timestamp_ns=np.int64(frame_ns),
                                point_time_mode=np.asarray(time_mode))

            point_ids = np.arange(0, len(p_world), max(1, math.ceil(len(p_world) / 120000)),
                                  dtype=np.int64)
            cam_good_count = 0
            for channel, files in camera_files.items():
                if not files:
                    sync_rows.append({"lidar_top_timestamp": frame_stamp, "channel": channel,
                                      "camera_timestamp": None, "image_path": None,
                                      "signed_camera_minus_lidar_ms": None,
                                      "image_reused": False, "suspected_freeze": False,
                                      "nav_pose_valid": False, "status": "missing_channel"})
                    continue
                cam_ns, image_path, signed_delta_ns = _select_camera(files, frame_ns)
                delta_ms = signed_delta_ns / 1e6
                stamps = [stamp for stamp, _ in files]
                cam_idx = int(np.searchsorted(stamps, cam_ns))
                gap_prev = (cam_ns - stamps[cam_idx - 1]) / 1e6 if cam_idx > 0 else None
                gap_next = (stamps[cam_idx + 1] - cam_ns) / 1e6 if cam_idx + 1 < len(stamps) else None
                reused = str(image_path) in used_images[channel]
                used_images[channel].add(str(image_path))
                is_freeze = cam_idx in freeze_by_stamp.get(channel, set())
                if is_freeze:
                    status = "suspected_freeze"
                elif abs(delta_ms) > max_sync_ms:
                    status = "invalid_delta"
                elif reused:
                    status = "reused_image"
                elif abs(delta_ms) <= good_sync_ms:
                    status = "good_sync"
                else:
                    status = "deskew_required"
                pose_cam = nav.at(cam_ns)
                nav_ok = bool(pose_cam.valid[0]) and pose_cam.heading_std_deg[0] <= 2.0
                if not nav_ok and status in ("good_sync", "deskew_required"):
                    status = "invalid_nav_pose"
                sync_rows.append({"lidar_top_timestamp": frame_stamp, "channel": channel,
                                  "camera_timestamp": image_path.stem, "image_path": str(image_path),
                                  "signed_camera_minus_lidar_ms": delta_ms,
                                  "camera_gap_prev_ms": gap_prev, "camera_gap_next_ms": gap_next,
                                  "image_reused": reused, "suspected_freeze": is_freeze,
                                  "nav_pose_valid": nav_ok,
                                  "nav_bracket_gap_ms": float(pose_cam.bracket_gap_ms[0]),
                                  "status": status})
                if status not in ("good_sync", "deskew_required"):
                    continue
                cam_pose = pose_cam
                r_cam_ego = cam_pose.rotation_world_ego[0].T
                p_ego_cam = (p_world[point_ids] - cam_pose.position_enu[0]) @ r_cam_ego
                t_cam_ego = camera_cal[channel]["ego_to_camera"]
                p_camera = p_ego_cam @ t_cam_ego[:3, :3].T + t_cam_ego[:3, 3]
                if visualization_stride > 0 and index % visualization_stride == 0:
                    try:
                        _overlay(image_path, p_camera, camera_cal[channel]["K"],
                                 camera_cal[channel]["dist"], output_dir / "visualizations",
                                 frame_stamp, channel, camera_cal[channel]["expected_size"])
                    except Exception as overlay_exc:
                        sync_rows[-1]["overlay_error"] = f"{type(overlay_exc).__name__}: {overlay_exc}"
                cam_good_count += 1
            frame_rows.append({"lidar_top_timestamp": frame_stamp, "lidar_path": str(lidar_path),
                               "point_count": len(p_top), "point_time_mode": time_mode,
                               "scan_point_time_min_ms": float((point_ns.min() - frame_ns) / 1e6),
                               "scan_point_time_max_ms": float((point_ns.max() - frame_ns) / 1e6),
                               "max_nav_bracket_gap_ms": float(max(np.max(point_pose.bracket_gap_ms),
                                                                    end_pose.bracket_gap_ms[0])),
                               "camera_views_usable": cam_good_count, "status": "ok"})
            success_count += 1
        except Exception as exc:  # Record a failed frame and continue the batch.
            (processed_dir / f"{frame_stamp}.npz").unlink(missing_ok=True)
            frame_rows.append({"lidar_top_timestamp": frame_stamp, "lidar_path": str(lidar_path),
                               "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

        if (index + 1) % 25 == 0 or index + 1 == len(lidar_files):
            print(f"Preprocessed LIDAR_TOP {index + 1}/{len(lidar_files)}; success={success_count}",
                  flush=True)

    _write_csv(output_dir / "camera_sync.csv", sync_rows)
    _write_csv(output_dir / "camera_quality.csv", camera_quality)
    _write_csv(output_dir / "camera_dropout_gaps.csv", dropout_rows)
    _write_csv(output_dir / "camera_freeze_candidates.csv", freeze_rows)
    _write_csv(output_dir / "lidar_top_deskew_manifest.csv", frame_rows)
    report = {
        "raw_session": str(raw_root), "output_dir": str(output_dir),
        "raw_data_modified": False, "lidar_source": "LIDAR_TOP",
        "lidar_frames": len(lidar_files), "lidar_frames_deskewed": success_count,
        "camera_channels": list(CAMERA_MAP.keys()), "camera_calibration_period": period_id,
        "calibration_date": session_date.isoformat(),
        "transform_contract": {
            "LIDAR_TOP": "Calib_2 T_EGO_FROM_LIDAR_TOP (sensor to ego)",
            "camera": "Calib_2 T_CAMERA_FROM_EGO (raw PNK camera matrix)",
            "ego_axes": "forward-left-up", "world_axes": "local ENU",
            "nav_yaw": "Heading2 compass clockwise from North -> ENU yaw = 90 deg - Heading2",
            "attitude": "NAV Pitch/Roll; Rz(yaw) Ry(-pitch) Rx(roll)",
            "gnss_lever_arm_applied": False,
        },
        "motion_compensation": {
            "point_timestamp_mode": point_time_mode,
            "cloud_output_frame": "ego at LIDAR_TOP scan-end timestamp",
            "camera_overlay": "each point motion-compensated to selected camera timestamp",
            "nav_max_bracket_gap_ms": max_nav_gap_ms,
            "deskew_assumes_static_world_points": True,
        },
        "sync_thresholds_ms": {"good": good_sync_ms, "maximum": max_sync_ms,
                                "dropout_gap": dropout_ms},
        "summary": {
            "usable_camera_sync_pairs": sum(r["status"] in ("good_sync", "deskew_required")
                                             for r in sync_rows),
            "invalid_or_suspect_camera_pairs": sum(r["status"] not in ("good_sync", "deskew_required")
                                                    for r in sync_rows),
            "camera_dropout_intervals": len(dropout_rows),
            "suspected_freeze_frames": len(freeze_rows),
            "failed_lidar_frames": len(lidar_files) - success_count,
        },
        "artifacts": ["camera_sync.csv", "camera_quality.csv", "camera_dropout_gaps.csv",
                      "camera_freeze_candidates.csv", "lidar_top_deskew_manifest.csv",
                      "lidar_top_deskewed_ego/", "visualizations/"],
        "limitations": [
            "GNSS-to-ego lever arm is not applied; verify it before high-accuracy turn deskew.",
            "NAV Pitch/Roll convention is recorded as an explicit assumption and needs visual validation.",
            "Point deskew assumes static scene points; independently moving objects can smear.",
            "Camera matrices use the ego-to-camera direction expected by the existing PNK adapter; validate overlays before calibration changes.",
            "This command audits calibration and timing; it does not modify Calib_2 or source labels.",
        ],
    }
    (output_dir / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown_report(output_dir / "preprocessing_report.md", report, camera_quality)
    return report


def _write_markdown_report(path: Path, report: dict, camera_quality: list[dict]) -> None:
    def number(value):
        return f"{value:.2f}" if isinstance(value, (int, float)) else "n/a"

    lines = ["# PNK LIDAR_TOP preprocessing report", "",
             f"- Raw session: `{report['raw_session']}`",
             f"- Output: `{report['output_dir']}`",
             f"- Data source: `{report['lidar_source']}`; raw data modified: **No**",
             f"- Calibration period: `{report['camera_calibration_period']}`",
             f"- Deskewed LiDAR_TOP frames: **{report['lidar_frames_deskewed']} / {report['lidar_frames']}**",
             f"- Camera usable sync pairs: **{report['summary']['usable_camera_sync_pairs']}**; invalid/suspect: **{report['summary']['invalid_or_suspect_camera_pairs']}**",
             f"- Camera dropout intervals: **{report['summary']['camera_dropout_intervals']}**; suspected freeze frames: **{report['summary']['suspected_freeze_frames']}**",
             "", "## Per-camera health", "",
             "| Camera | Images | Median period (ms) | P95 (ms) | Max gap (ms) | Dropouts | Freeze candidates | Status |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in camera_quality:
        lines.append(f"| {row['channel']} | {row['image_count']} | {number(row['median_period_ms'])} | "
                     f"{number(row['p95_period_ms'])} | {number(row['max_gap_ms'])} | "
                     f"{row['dropout_gaps_over_threshold']} | {row['suspected_freeze_frames']} | "
                     f"{row.get('status', 'available')} |")
    lines.extend(["", "## Method and cautions", "",
                  "Point-wise ego-motion compensation uses NAV position, `Heading2`, `Pitch`, and `Roll`; "
                  "point clouds are written in the ego frame at scan end. Camera overlays compensate "
                  "each return to the selected image timestamp before projection.", "",
                  "The top LiDAR transform is interpreted as `T_EGO_FROM_LIDAR_TOP`; camera matrices "
                  "as `T_CAMERA_FROM_EGO`. Check the overlays before accepting this convention or "
                  "changing any calibration. NAV-to-ego lever arm is not applied. Dynamic objects are "
                  "not motion compensated, so use static scene geometry to review calibration.", "",
                  "See `camera_sync.csv`, `camera_dropout_gaps.csv`, `camera_freeze_candidates.csv`, "
                  "`lidar_top_deskew_manifest.csv`, and `visualizations/` for per-frame evidence.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only LIDAR_TOP sync/calibration preprocessing with NAV deskew")
    parser.add_argument("--raw", type=Path, required=True, help="One PNK raw session folder")
    parser.add_argument("--calibration", type=Path, required=True, help="PNKData_meta_fresh/Calib_2")
    parser.add_argument("--out", type=Path, required=True, help="Derived output folder outside --raw")
    parser.add_argument("--point-time-mode", choices=("auto", "epoch_ns", "offset_ns", "offset_s"), default="auto")
    parser.add_argument("--limit", type=int, help="Process only the first N LIDAR_TOP frames")
    parser.add_argument("--visualization-stride", type=int, default=10,
                        help="Write raw/undistorted projection overlays every N LiDAR frames; 0 disables")
    parser.add_argument("--good-sync-ms", type=float, default=DEFAULT_GOOD_SYNC_MS)
    parser.add_argument("--max-sync-ms", type=float, default=DEFAULT_MAX_SYNC_MS)
    parser.add_argument("--dropout-ms", type=float, default=DEFAULT_DROPOUT_MS)
    parser.add_argument("--max-nav-gap-ms", type=float, default=DEFAULT_MAX_NAV_GAP_MS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = preprocess_session(args.raw, args.calibration, args.out,
                                point_time_mode=args.point_time_mode, limit=args.limit,
                                visualization_stride=args.visualization_stride,
                                good_sync_ms=args.good_sync_ms, max_sync_ms=args.max_sync_ms,
                                dropout_ms=args.dropout_ms, max_nav_gap_ms=args.max_nav_gap_ms)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
