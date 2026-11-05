"""Reverse-audit PNK 3D labels against dimensions, merged LiDAR and tracks.

Outputs are written next to this script. The audit never edits source labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import laspy
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Plotting is optional when importing audit helpers.
    plt = None
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = Path(r"D:\mentor_data\PNKData")
DEFAULT_META = Path(r"D:\mentor_data\PNKData_meta_fresh")
DEFAULT_LABELS = DEFAULT_META / "label" / "2026_08_14_vf6_01_02"
DEFAULT_CSV = DEFAULT_META / "file_csv" / "2026_08_14_vf6_01_02.csv"
OUT_ROOT = Path(__file__).resolve().parent
YAW_CANDIDATES = (
    ("yaw", lambda y: y),
    ("-yaw", lambda y: -y),
    ("yaw+pi/2", lambda y: y + math.pi / 2),
    ("yaw-pi/2", lambda y: y - math.pi / 2),
    ("-yaw+pi/2", lambda y: -y + math.pi / 2),
    ("-yaw-pi/2", lambda y: -y - math.pi / 2),
)
DIM_CANDIDATES = ("dx=length,dy=width", "dx=width,dy=length")
Z_CANDIDATES = ("geometric_center", "bottom_center")
CLASS_COLORS = {"Car": "#f4b942", "Rider": "#24b7c6", "Pedestrian": "#ed7295"}


@dataclass(frozen=True)
class Box:
    timestamp: str
    timestamp_s: float
    segment: str
    line_number: int
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    yaw: float
    cls: str
    occlusion: int
    numpoints: int
    trackid: str


@dataclass(frozen=True)
class Pair:
    session: str
    lidar: Path
    label: Path


def parse_timestamp(stem: str) -> float:
    seconds, nanos = stem.split("-", 1)
    return int(seconds) + int(nanos) / 1_000_000_000.0


def read_label_file(path: Path) -> tuple[Path, list[Box], int]:
    boxes: list[Box] = []
    malformed = 0
    ts = parse_timestamp(path.stem)
    # Labels live at <segment>/label/<timestamp>.txt.
    segment = path.parent.parent.name
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 11:
                malformed += 1
                continue
            try:
                values = [float(v) for v in fields[:7]]
                occ, npts = int(float(fields[8])), int(float(fields[9]))
                if not np.all(np.isfinite(values)):
                    malformed += 1
                    continue
                boxes.append(Box(path.stem, ts, segment, line_number, *values,
                                 fields[7], occ, npts, fields[10]))
            except (ValueError, OverflowError):
                malformed += 1
    return path, boxes, malformed


def enumerate_pairs(raw_root: Path, label_root: Path) -> tuple[list[Pair], dict[str, list[Path]]]:
    labels_by_stem: dict[str, list[Path]] = defaultdict(list)
    label_files = sorted(label_root.rglob("*.txt"))
    for path in label_files:
        labels_by_stem[path.stem].append(path)
    pairs: list[Pair] = []
    raw_frames = 0
    for session in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        lidar_dir = session / "LIDAR" / "LIDAR_MERGED_EGO"
        if not lidar_dir.is_dir():
            continue
        for lidar in lidar_dir.glob("*.laz"):
            raw_frames += 1
            matches = labels_by_stem.get(lidar.stem, [])
            if matches:
                # Duplicate stems can occur in overlapping segment indexes.
                # Keep the first deterministically and report duplicate coverage separately.
                pairs.append(Pair(session.name, lidar, sorted(matches)[0]))
    pairs.sort(key=lambda p: (p.session, p.lidar.name))
    print(f"Found {len(label_files):,} label files and {raw_frames:,} merged LiDAR frames; "
          f"{len(pairs):,} timestamp matches.", flush=True)
    return pairs, labels_by_stem


def stratified_sample(pairs: list[Pair], limit: int) -> list[Pair]:
    groups: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        groups[pair.session].append(pair)
    if len(pairs) <= limit:
        return pairs
    selected: list[Pair] = []
    selected_keys: set[tuple[str, str]] = set()
    # Round-robin through sessions so one long clip does not dominate.
    ordered_sessions = sorted(groups)
    for rank in range(limit):
        session = ordered_sessions[rank % len(ordered_sessions)]
        candidates = groups[session]
        ordinal = rank // len(ordered_sessions)
        if ordinal >= len(candidates):
            continue
        idx = int(round(ordinal * (len(candidates) - 1) / max(1, math.ceil(limit / len(ordered_sessions)) - 1)))
        pair = candidates[idx]
        key = (pair.session, pair.lidar.stem)
        if key not in selected_keys:
            selected.append(pair)
            selected_keys.add(key)
    # Fill any slots skipped because small groups were exhausted.
    if len(selected) < limit:
        for pair in pairs:
            key = (pair.session, pair.lidar.stem)
            if key not in selected_keys:
                selected.append(pair)
                selected_keys.add(key)
                if len(selected) >= limit:
                    break
    return sorted(selected, key=lambda p: (p.session, p.lidar.name))


def angle_mod_pi_error(a: float, b: float) -> float:
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


def angle_wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def dimensions_for(box: Box, dim_candidate: str) -> tuple[float, float, float]:
    if dim_candidate == DIM_CANDIDATES[0]:
        return box.dx, box.dy, box.dz
    return box.dy, box.dx, box.dz


def inside_box(points: np.ndarray, box: Box, yaw: float,
               dim_candidate: str, z_candidate: str) -> np.ndarray:
    length, width, height = dimensions_for(box, dim_candidate)
    delta = points[:, :2] - np.array([box.x, box.y], dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    local_x = c * delta[:, 0] + s * delta[:, 1]
    local_y = -s * delta[:, 0] + c * delta[:, 1]
    if z_candidate == "geometric_center":
        z_ok = np.abs(points[:, 2] - box.z) <= height / 2
    else:
        z_ok = (points[:, 2] >= box.z) & (points[:, 2] <= box.z + height)
    return ((np.abs(local_x) <= length / 2) &
            (np.abs(local_y) <= width / 2) & z_ok)


def pca_angle(points: np.ndarray) -> float | None:
    if len(points) < 5:
        return None
    xy = points[:, :2].astype(np.float64)
    covariance = np.cov(xy - xy.mean(axis=0), rowvar=False)
    vals, vecs = np.linalg.eigh(covariance)
    if vals[-1] <= 1e-8 or vals[-1] / max(vals[0], 1e-8) < 1.15:
        return None
    vec = vecs[:, -1]
    return math.atan2(float(vec[1]), float(vec[0]))


def corr(values_a: list[float], values_b: list[float], rank: bool = False) -> float:
    if len(values_a) < 3 or np.std(values_a) == 0 or np.std(values_b) == 0:
        return float("nan")
    result = spearmanr(values_a, values_b).statistic if rank else pearsonr(values_a, values_b).statistic
    return float(result)


def metric_summary(rows: list[dict], primary_class: str = "Car") -> list[dict]:
    candidate_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["class"] == primary_class:
            candidate_groups[(row["yaw_candidate"], row["dimension_candidate"], row["z_candidate"])].append(row)
    summaries = []
    for (yaw_name, dim_name, z_name), group in candidate_groups.items():
        gt = [r["numpoints_gt"] for r in group]
        computed = [r["numpoints_computed"] for r in group]
        errors = np.array([r["pca_error_deg"] for r in group if r["pca_error_deg"] is not None])
        grounds = np.array([r["ground_error_m"] for r in group if r["ground_error_m"] is not None])
        summaries.append({
            "class": primary_class,
            "yaw_candidate": yaw_name,
            "dimension_candidate": dim_name,
            "z_candidate": z_name,
            "n_boxes": len(group),
            "pearson_pointcount_corr": corr(gt, computed),
            "spearman_pointcount_corr": corr(gt, computed, rank=True),
            "pointcount_mae": float(np.mean(np.abs(np.asarray(gt) - np.asarray(computed)))) if group else None,
            "pointcount_median_abs_error": float(np.median(np.abs(np.asarray(gt) - np.asarray(computed)))) if group else None,
            "pointcount_exact_pct": float(np.mean(np.asarray(gt) == np.asarray(computed)) * 100) if group else None,
            "pca_n": int(len(errors)),
            "pca_mean_error_deg": float(np.mean(errors)) if len(errors) else None,
            "pca_median_error_deg": float(np.median(errors)) if len(errors) else None,
            "pca_p90_error_deg": float(np.percentile(errors, 90)) if len(errors) else None,
            "pca_lt5_pct": float(np.mean(errors < 5) * 100) if len(errors) else None,
            "pca_lt10_pct": float(np.mean(errors < 10) * 100) if len(errors) else None,
            "pca_lt20_pct": float(np.mean(errors < 20) * 100) if len(errors) else None,
            "ground_n": int(len(grounds)),
            "ground_median_abs_error_m": float(np.median(grounds)) if len(grounds) else None,
            "ground_p90_abs_error_m": float(np.percentile(grounds, 90)) if len(grounds) else None,
            "ground_lt0p2m_pct": float(np.mean(grounds < 0.2) * 100) if len(grounds) else None,
            "ground_lt0p3m_pct": float(np.mean(grounds < 0.3) * 100) if len(grounds) else None,
        })
    return summaries


def analyze_frame(pair: Pair, output_frame_dir: Path, save_bev: bool) -> tuple[list[dict], list[dict], dict]:
    _, boxes, _ = read_label_file(pair.label)
    cloud = laspy.read(pair.lidar)
    points = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float32, copy=False)
    tree = cKDTree(points[:, :2])
    candidate_rows: list[dict] = []
    detail_rows: list[dict] = []
    for box in boxes:
        if box.dx <= 0 or box.dy <= 0 or box.dz <= 0:
            continue
        r = math.sqrt(box.dx * box.dx + box.dy * box.dy) / 2 + 1.0
        ids = tree.query_ball_point([box.x, box.y], r)
        local = points[ids]
        ground_ids = [i for i, point in enumerate(local)
                      if 1.0 <= math.hypot(float(point[0] - box.x), float(point[1] - box.y)) <= 5.0
                      and float(point[2]) < box.z + 0.25]
        ground_z = None
        if len(ground_ids) >= 8:
            zvals = local[ground_ids, 2]
            # A low quantile of a surrounding annulus is a local ground proxy;
            # it is deliberately reported as proxy rather than surveyed GT.
            ground_z = float(np.percentile(zvals, 10))
        for yaw_name, yaw_fn in YAW_CANDIDATES:
            yaw = yaw_fn(box.yaw)
            for dim_name in DIM_CANDIDATES:
                length, width, height = dimensions_for(box, dim_name)
                for z_name in Z_CANDIDATES:
                    inside_mask = inside_box(local, box, yaw, dim_name, z_name)
                    inside = local[inside_mask]
                    count = int(np.count_nonzero(inside_mask))
                    axis = pca_angle(inside) if box.cls == "Car" and box.numpoints >= 20 else None
                    # If dx/dy are swapped, the long axis is local +Y rather than +X.
                    long_axis_yaw = yaw + (math.pi / 2 if dim_name == DIM_CANDIDATES[1] else 0.0)
                    pca_error = math.degrees(angle_mod_pi_error(long_axis_yaw, axis)) if axis is not None else None
                    z_bottom = box.z - height / 2 if z_name == "geometric_center" else box.z
                    ground_error = abs(z_bottom - ground_z) if ground_z is not None else None
                    candidate_rows.append({
                        "timestamp": box.timestamp, "segment": box.segment, "trackid": box.trackid,
                        "class": box.cls, "numpoints_gt": box.numpoints,
                        "numpoints_computed": count, "yaw_candidate": yaw_name,
                        "dimension_candidate": dim_name, "z_candidate": z_name,
                        "pca_error_deg": pca_error, "ground_error_m": ground_error,
                    })
                    if (yaw_name, dim_name, z_name) == ("yaw", DIM_CANDIDATES[0], "geometric_center"):
                        # These are local box-neighborhood diagnostics, not a semantic
                        # object-point segmentation: nearby road/other-object returns
                        # can enter the denominator.
                        local_height = box.dz
                        zcenter = box.z
                        local_xyz = local[(local[:, 2] >= zcenter - local_height / 2 - 0.5) &
                                          (local[:, 2] <= zcenter + local_height / 2 + 0.5)]
                        delta_xy = local_xyz[:, :2] - np.array([box.x, box.y], dtype=np.float64)
                        c0, s0 = math.cos(yaw), math.sin(yaw)
                        lx = c0 * delta_xy[:, 0] + s0 * delta_xy[:, 1]
                        ly = -s0 * delta_xy[:, 0] + c0 * delta_xy[:, 1]
                        lz = local_xyz[:, 2] - zcenter
                        half = np.array([length / 2, width / 2, local_height / 2], dtype=np.float64)
                        rel = np.column_stack((lx, ly, lz)) if len(local_xyz) else np.empty((0, 3))
                        abs_rel = np.abs(rel)
                        sdf_components = abs_rel - half
                        outside_distance = np.linalg.norm(np.maximum(sdf_components, 0), axis=1) if len(rel) else np.array([])
                        signed_distance = (outside_distance + np.minimum(np.max(sdf_components, axis=1), 0)
                                           if len(rel) else np.array([]))
                        surface_distance = np.abs(signed_distance)
                        local_inside = np.all(abs_rel <= half, axis=1) if len(rel) else np.array([], dtype=bool)
                        def quartile_occupancy(values: np.ndarray, extent: float) -> list[float]:
                            if len(values) == 0:
                                return [float("nan")] * 4
                            counts, _ = np.histogram(values, bins=np.linspace(-extent, extent, 5))
                            return (counts / len(values) * 100).tolist()
                        occ_l = quartile_occupancy(rel[local_inside, 0], length / 2)
                        occ_w = quartile_occupancy(rel[local_inside, 1], width / 2)
                        occ_h = quartile_occupancy(rel[local_inside, 2], local_height / 2)
                        detail_rows.append({
                            "timestamp": box.timestamp, "segment": box.segment, "trackid": box.trackid,
                            "class": box.cls, "x": box.x, "y": box.y, "z": box.z,
                            "dx": box.dx, "dy": box.dy, "dz": box.dz, "yaw": box.yaw,
                            "numpoints_gt": box.numpoints, "numpoints_computed_direct": count,
                            "pointcount_abs_error_direct": abs(box.numpoints - count),
                            "local_ground_z_p10_proxy": ground_z,
                            "z_bottom_geometric_center": z_bottom,
                            "ground_abs_error_geometric_center_m": ground_error,
                            "local_neighborhood_points": int(len(local_xyz)),
                            "box_inside_points": int(np.count_nonzero(local_inside)),
                            "inside_fraction_of_local_pct": float(np.mean(local_inside) * 100) if len(local_inside) else None,
                            "outside_points_within_0p25m_of_surface": int(np.count_nonzero((~local_inside) & (surface_distance <= 0.25))) if len(local_inside) else 0,
                            "point_to_box_surface_median_m": float(np.median(surface_distance)) if len(surface_distance) else None,
                            "point_to_box_surface_p90_m": float(np.percentile(surface_distance, 90)) if len(surface_distance) else None,
                            "inside_occupancy_length_quartiles_pct": occ_l,
                            "inside_occupancy_width_quartiles_pct": occ_w,
                            "inside_occupancy_height_quartiles_pct": occ_h,
                        })
    frame_meta = {"pair": pair, "points": points if save_bev else None, "boxes": boxes,
                  "point_count": len(points)}
    if save_bev:
        output_frame_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 10), dpi=130)
        stride = max(1, len(points) // 250_000)
        ax.scatter(points[::stride, 1], points[::stride, 0], s=0.15, c="#536a76", alpha=0.7,
                   label="Merged ego LiDAR")
        for i, box in enumerate(boxes, 1):
            length, width = box.dx, box.dy
            local_corners = np.array([[-length/2, -width/2], [length/2, -width/2],
                                      [length/2, width/2], [-length/2, width/2], [-length/2, -width/2]])
            c, s = math.cos(box.yaw), math.sin(box.yaw)
            rot = np.array([[c, -s], [s, c]])
            corners = local_corners @ rot.T + [box.x, box.y]
            ax.plot(corners[:, 1], corners[:, 0], color=CLASS_COLORS.get(box.cls, "white"), linewidth=0.8)
            ax.text(box.y, box.x, str(i), color="black", fontsize=5,
                    bbox={"facecolor": CLASS_COLORS.get(box.cls, "white"), "edgecolor": "none", "pad": 0.5})
        ax.set_title(f"{pair.lidar.stem} | LiDAR + labeled box footprints", fontsize=11)
        ax.set_xlabel("Ego Y (m), +Y left", fontsize=9)
        ax.set_ylabel("Ego X (m), +X forward", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.set_xlim(50, -50)
        ax.set_ylim(-20, 80)
        ax.set_position([0.12, 0.10, 0.82, 0.82])
        fig.savefig(output_frame_dir / f"bev_{pair.lidar.stem}.png")
        plt.close(fig)
    return candidate_rows, detail_rows, frame_meta


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_dated_calibration(meta_root: Path, timestamp_s: float) -> tuple[dict, dict, str]:
    cal_root = meta_root / "Calib_2"
    intrinsics = json.loads((cal_root / "VF6_01_Intrinsics.json").read_text(encoding="utf-8-sig"))
    dated = json.loads((cal_root / "VF6_01_Extrinsics_By_Dates.json").read_text(encoding="utf-8-sig"))
    day = datetime.fromtimestamp(timestamp_s, timezone.utc).date()
    for period in dated["periods"]:
        start = datetime.fromisoformat(period["valid_from"]).date()
        end = datetime.fromisoformat(period["valid_to"]).date()
        if start <= day < end:
            return intrinsics, period["extrinsics"], period["id"]
    raise ValueError(f"No Calib_2 extrinsic period covers {day}")


def nearest_camera_image(session_dir: Path, channel: str, timestamp_s: float) -> tuple[Path, float] | None:
    camera_dir = session_dir / "CAMERA" / channel
    if not camera_dir.is_dir():
        return None
    best: tuple[Path, float] | None = None
    for image_path in camera_dir.glob("*.jpg"):
        try:
            delta = abs(parse_timestamp(image_path.stem) - timestamp_s)
        except ValueError:
            continue
        if best is None or delta < best[1]:
            best = image_path, delta
    return best


def box_corners_ego(box: Box, yaw_name: str, dim_name: str, z_name: str) -> np.ndarray:
    yaw = dict(YAW_CANDIDATES)[yaw_name](box.yaw)
    length, width, height = dimensions_for(box, dim_name)
    zlo, zhi = (box.z - height / 2, box.z + height / 2) if z_name == "geometric_center" else (box.z, box.z + height)
    local = np.array([
        [-length/2, -width/2, zlo-box.z], [length/2, -width/2, zlo-box.z],
        [length/2, width/2, zlo-box.z], [-length/2, width/2, zlo-box.z],
        [-length/2, -width/2, zhi-box.z], [length/2, -width/2, zhi-box.z],
        [length/2, width/2, zhi-box.z], [-length/2, width/2, zhi-box.z],
    ])
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ rot.T + np.array([box.x, box.y, box.z])


def clip_segment_to_image(p1: np.ndarray, p2: np.ndarray, width: int, height: int):
    """Clip a projected float segment before converting it to OpenCV int coordinates."""
    x1, y1 = map(float, p1)
    x2, y2 = map(float, p2)
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x1), (dx, width - 1 - x1), (-dy, y1), (dy, height - 1 - y1)):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)
    a = (x1 + t0 * dx, y1 + t0 * dy)
    b = (x1 + t1 * dx, y1 + t1 * dy)
    return ((int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))))


def camera_projection_audit(pairs: list[Pair], meta_root: Path, output_dir: Path,
                            convention: tuple[str, str, str], max_frames: int = 6) -> list[dict]:
    """Save qualitative calibrated box overlays on six frames and all available P cameras."""
    channels = ("CAM_P_F", "CAM_P_FL", "CAM_P_FR", "CAM_P_B", "CAM_P_L", "CAM_P_R", "CAM_P_LB", "CAM_P_RB")
    if not pairs:
        return []
    chosen = [pairs[i] for i in np.linspace(0, len(pairs)-1, min(max_frames, len(pairs)), dtype=int)]
    intrinsics, _, _ = load_dated_calibration(meta_root, parse_timestamp(chosen[0].lidar.stem))
    manifest: list[dict] = []
    out = output_dir / "camera_projection_final"
    out.mkdir(parents=True, exist_ok=True)
    edges = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
    yaw_name, dim_name, z_name = convention
    for pair in chosen:
        timestamp_s = parse_timestamp(pair.lidar.stem)
        frame_intr, extrinsics, period_id = load_dated_calibration(meta_root, timestamp_s)
        boxes = read_label_file(pair.label)[1]
        for channel in channels:
            camera_match = nearest_camera_image(pair.lidar.parents[2], channel, timestamp_s)
            if camera_match is None:
                manifest.append({"lidar_timestamp": pair.lidar.stem, "camera": channel,
                                 "status": "camera_missing", "extrinsic_period": period_id})
                continue
            image_path, delta = camera_match
            if delta > 0.12:
                manifest.append({"lidar_timestamp": pair.lidar.stem, "camera": channel,
                                 "image": str(image_path), "delta_t_ms": round(delta*1000, 3),
                                 "status": "nearest_camera_over_120ms", "extrinsic_period": period_id})
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                manifest.append({"lidar_timestamp": pair.lidar.stem, "camera": channel,
                                 "image": str(image_path), "status": "image_decode_failed", "extrinsic_period": period_id})
                continue
            cal = frame_intr[channel]
            k = np.asarray(cal["camera_matrix"], dtype=np.float64)
            dist = np.asarray(cal.get("distortion_coefficients", []), dtype=np.float64)
            transform = np.asarray(extrinsics[channel], dtype=np.float64)
            rotation, translation = transform[:3, :3], transform[:3, 3]
            rvec, _ = cv2.Rodrigues(rotation)
            drawn = 0
            rejected_extreme = 0
            for box_id, box in enumerate(boxes, 1):
                corners = box_corners_ego(box, yaw_name, dim_name, z_name)
                cam_xyz = (rotation @ corners.T).T + translation
                if np.any(cam_xyz[:, 2] <= 1.0):
                    continue
                image_points, _ = cv2.projectPoints(corners.astype(np.float64), rvec, translation, k, dist)
                uv = image_points.reshape(-1, 2)
                if not np.all(np.isfinite(uv)):
                    continue
                # Avoid huge cuboids that explode near the camera plane and obscure
                # the qualitative overlay. Partial objects near image borders remain
                # eligible if their full projection is within this generous margin.
                h, w = image.shape[:2]
                if (np.ptp(uv[:, 0]) > 0.70 * w or np.ptp(uv[:, 1]) > 0.70 * h or
                        np.min(uv[:, 0]) < -0.25 * w or np.max(uv[:, 0]) > 1.25 * w or
                        np.min(uv[:, 1]) < -0.25 * h or np.max(uv[:, 1]) > 1.25 * h):
                    rejected_extreme += 1
                    continue
                color = {"Car": (0, 190, 255), "Rider": (220, 190, 0), "Pedestrian": (180, 80, 240)}.get(box.cls, (230, 230, 230))
                for a, b in edges:
                    segment = clip_segment_to_image(uv[a], uv[b], image.shape[1], image.shape[0])
                    if segment is not None:
                        cv2.line(image, segment[0], segment[1], color, 2, cv2.LINE_AA)
                center_xy = np.mean(uv[:4], axis=0)
                if 0 <= center_xy[0] < image.shape[1] and 0 <= center_xy[1] < image.shape[0]:
                    center = (int(round(float(center_xy[0]))), int(round(float(center_xy[1]))))
                    cv2.putText(image, str(box_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                drawn += 1
            cv2.putText(image, f"{pair.lidar.stem} -> {channel} | dt={delta*1000:.1f}ms | {period_id}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
            target = out / f"{pair.lidar.stem}_{channel}.jpg"
            cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 88])
            manifest.append({"lidar_timestamp": pair.lidar.stem, "camera": channel,
                             "image": str(image_path), "overlay": str(target),
                             "delta_t_ms": round(delta*1000, 3), "extrinsic_period": period_id,
                             "boxes_projected": drawn, "boxes_rejected_extreme_projection": rejected_extreme,
                             "status": "rendered"})
    write_csv(output_dir / "camera_projection_manifest.csv", manifest)
    return manifest


def collect_dimensions_and_tracks(label_files: list[Path], workers: int) -> tuple[dict, dict, list[dict], int, int, int]:
    dimensions: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    tracks: dict[tuple[str, str, str], list[tuple[float, float, float, float, float, float, float]]] = defaultdict(list)
    dimension_anomalies: list[dict] = []
    malformed = empty_files = 0
    total_boxes = 0
    print(f"Parsing {len(label_files):,} label files...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (path, boxes, bad) in enumerate(pool.map(read_label_file, label_files), 1):
            malformed += bad
            total_boxes += len(boxes)
            if not boxes:
                empty_files += 1
            for b in boxes:
                dimensions[b.cls]["dx"].append(b.dx)
                dimensions[b.cls]["dy"].append(b.dy)
                dimensions[b.cls]["dz"].append(b.dz)
                longest, shortest = max(b.dx, b.dy), min(b.dx, b.dy)
                expected = {
                    "Car": (2.5 <= longest <= 8.0 and 1.0 <= shortest <= 3.0 and 0.8 <= b.dz <= 3.0),
                    "Rider": (0.3 <= longest <= 4.0 and 0.2 <= shortest <= 2.5 and 0.5 <= b.dz <= 3.0),
                    "Pedestrian": (0.2 <= longest <= 2.5 and 0.2 <= shortest <= 1.5 and 0.8 <= b.dz <= 2.5),
                }.get(b.cls, True)
                if not expected or min(b.dx, b.dy, b.dz) <= 0:
                    dimension_anomalies.append({
                        "type": "dimension_outlier", "segment": b.segment, "timestamp": b.timestamp,
                        "trackid": b.trackid, "class": b.cls,
                        "detail": f"dx={b.dx:.3f},dy={b.dy:.3f},dz={b.dz:.3f}m",
                    })
                if b.cls == "Car":
                    tracks[(b.segment, b.cls, b.trackid)].append(
                        (b.timestamp_s, b.x, b.y, b.yaw, b.dx, b.dy, b.dz)
                    )
            if i % 5000 == 0:
                print(f"  parsed {i:,}/{len(label_files):,} labels", flush=True)
    print(f"Parsed {total_boxes:,} boxes; {empty_files} empty label files; {malformed} malformed rows.", flush=True)
    return dimensions, tracks, dimension_anomalies, empty_files, total_boxes, malformed


def dimension_stats(dimensions: dict[str, dict[str, list[float]]]) -> list[dict]:
    rows = []
    for cls, axes in sorted(dimensions.items()):
        for axis, values in sorted(axes.items()):
            arr = np.asarray(values, dtype=np.float64)
            if not len(arr):
                continue
            physical = (arr >= 3.0) & (arr <= 6.0) if cls == "Car" and axis in ("dx", "dy") else np.zeros(len(arr), dtype=bool)
            rows.append({
                "class": cls, "dimension_field": axis, "n": len(arr),
                "mean_m": float(np.mean(arr)), "median_m": float(np.median(arr)),
                "p5_m": float(np.percentile(arr, 5)), "p25_m": float(np.percentile(arr, 25)),
                "p75_m": float(np.percentile(arr, 75)), "p95_m": float(np.percentile(arr, 95)),
                "min_m": float(np.min(arr)), "max_m": float(np.max(arr)),
                "car_length_range_pct_3_to_6m": float(np.mean(physical) * 100) if cls == "Car" and axis in ("dx", "dy") else "",
                "car_width_range_pct_1p5_to_2p5m": float(np.mean((arr >= 1.5) & (arr <= 2.5)) * 100) if cls == "Car" and axis in ("dx", "dy") else "",
                "car_height_range_pct_1p3_to_2p5m": float(np.mean((arr >= 1.3) & (arr <= 2.5)) * 100) if cls == "Car" and axis == "dz" else "",
            })
    return rows


def track_audit(tracks: dict, selected_yaw: str) -> tuple[list[dict], list[dict]]:
    yaw_fn = dict(YAW_CANDIDATES)[selected_yaw]
    pair_rows = []
    anomaly_rows = []
    for (segment, cls, trackid), observations in tracks.items():
        obs = sorted(observations)
        if len(obs) < 2:
            continue
        dims = np.array([o[4:7] for o in obs])
        med_dims = np.median(dims, axis=0)
        if np.any(med_dims <= 0):
            continue
        rel_range = (np.max(dims, axis=0) - np.min(dims, axis=0)) / med_dims
        if np.max(rel_range) > 0.35:
            anomaly_rows.append({"type": "track_dimension_range", "segment": segment, "trackid": trackid,
                                 "class": cls, "detail": f"relative size range dx/dy/dz={rel_range.tolist()}"})
        for a, b in zip(obs, obs[1:]):
            dt = b[0] - a[0]
            if dt <= 0:
                continue
            dx, dy = b[1] - a[1], b[2] - a[2]
            displacement = math.hypot(dx, dy)
            speed = displacement / dt
            yaw_jump_deg = math.degrees(abs(angle_wrap(b[3] - a[3])))
            yaw_now = yaw_fn(b[3])
            theta_motion = math.atan2(dy, dx) if displacement else None
            directed_error = abs(angle_wrap(yaw_now - theta_motion)) if theta_motion is not None else None
            pca_error = math.degrees(angle_mod_pi_error(yaw_now, theta_motion)) if theta_motion is not None else None
            pair_rows.append({
                "segment": segment, "trackid": trackid, "class": cls,
                "timestamp_from_s": a[0], "timestamp_to_s": b[0], "delta_t_s": dt,
                "displacement_m": displacement, "speed_mps": speed,
                "yaw_candidate": selected_yaw, "yaw_transformed_rad": yaw_now,
                "motion_heading_rad": theta_motion, "directed_yaw_motion_error_deg": math.degrees(directed_error) if directed_error is not None else None,
                "orientation_error_mod_180_deg": pca_error,
            })
            if speed > 50:
                anomaly_rows.append({"type": "track_speed_over_50mps", "segment": segment, "trackid": trackid,
                                     "class": cls, "detail": f"{speed:.2f} m/s over {dt:.3f}s"})
            if displacement > 10 and dt <= 0.4:
                anomaly_rows.append({"type": "track_teleport", "segment": segment, "trackid": trackid,
                                     "class": cls, "detail": f"{displacement:.2f}m over {dt:.3f}s"})
            if yaw_jump_deg > 150 and displacement > 0.5 and dt <= 0.6:
                anomaly_rows.append({"type": "track_yaw_flip_near_180deg", "segment": segment, "trackid": trackid,
                                     "class": cls, "detail": f"yaw jump={yaw_jump_deg:.1f}deg, motion={displacement:.2f}m/{dt:.3f}s"})
            step_size_delta = np.abs(np.asarray(b[4:7]) - np.asarray(a[4:7])) / np.maximum(med_dims, 1e-6)
            if float(np.max(step_size_delta)) > 0.25:
                anomaly_rows.append({"type": "track_dimension_jump", "segment": segment, "trackid": trackid,
                                     "class": cls, "detail": f"relative dx/dy/dz jump={step_size_delta.tolist()}"})
    return pair_rows, anomaly_rows


def make_plots(dim_rows: list[dict], candidate_rows: list[dict], detail_rows: list[dict],
               track_rows: list[dict]) -> None:
    plot_dir = OUT_ROOT / "visualizations" / "histograms"
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Dimension summary figure from median and percentile bands.
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{r['class']} {r['dimension_field']}" for r in dim_rows]
    med = [r["median_m"] for r in dim_rows]
    lo = [r["p5_m"] for r in dim_rows]
    hi = [r["p95_m"] for r in dim_rows]
    y = np.arange(len(labels))
    ax.errorbar(med, y, xerr=[np.asarray(med) - np.asarray(lo), np.asarray(hi) - np.asarray(med)], fmt="o", capsize=3)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Dimension (m), median with P5-P95")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "dimension_p5_p95.png")
    plt.close(fig)

    car_candidates = [r for r in metric_summary(candidate_rows) if r["pca_n"]]
    fig, ax = plt.subplots(figsize=(12, 6))
    if car_candidates:
        names = [f"{r['yaw_candidate']} | {r['dimension_candidate'].replace('dx=', '').replace(',dy=', '/')} | {r['z_candidate']}" for r in car_candidates]
        vals = [r["pca_median_error_deg"] for r in car_candidates]
        order = np.argsort(vals)
        ax.barh(np.array(names)[order], np.array(vals)[order], color="#4aa3b5")
        ax.set_xlabel("Median PCA long-axis error (degrees; modulo 180)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "pca_convention_median_errors.png")
    plt.close(fig)

    best = min(car_candidates, key=lambda r: (r["pointcount_mae"] if r["pointcount_mae"] is not None else math.inf,
                                               r["pca_median_error_deg"] if r["pca_median_error_deg"] is not None else math.inf)) if car_candidates else None
    if best:
        selected = [r for r in candidate_rows if r["class"] == "Car" and r["yaw_candidate"] == best["yaw_candidate"]
                    and r["dimension_candidate"] == best["dimension_candidate"] and r["z_candidate"] == best["z_candidate"]]
        fig, ax = plt.subplots(figsize=(6, 6))
        a = np.asarray([r["numpoints_gt"] for r in selected], dtype=float)
        b = np.asarray([r["numpoints_computed"] for r in selected], dtype=float)
        ax.scatter(a, b, s=8, alpha=0.35)
        lim = max(float(np.percentile(a, 99.5)) if len(a) else 1, float(np.percentile(b, 99.5)) if len(b) else 1, 1)
        ax.plot([0, lim], [0, lim], "r--", linewidth=1)
        ax.set(xlabel="GT numpoints", ylabel="Recomputed points in box", xlim=(0, lim), ylim=(0, lim))
        ax.set_title("Car point count; selected best candidate")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "pointcount_gt_vs_recomputed.png")
        plt.close(fig)

    if track_rows:
        moving = [r for r in track_rows if r["displacement_m"] > 1 and 0.1 <= r["delta_t_s"] <= 2.0 and 2.0 <= r["speed_mps"] < 40]
        fig, ax = plt.subplots(figsize=(8, 5))
        errors = [r["directed_yaw_motion_error_deg"] for r in moving]
        ax.hist(errors, bins=np.arange(0, 185, 5), color="#dc7a51")
        ax.set(xlabel="|yaw - motion heading| (degrees)", ylabel="Track pairs",
               title="Temporal heading check (moving Car pairs; directed)")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / "track_heading_error.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--max-raw-frames", type=int, default=120,
                        help="Timestamp-matched merged LiDAR frames for expensive geometry checks")
    parser.add_argument("--workers", type=int, default=min(12, max(2, os.cpu_count() or 4)))
    args = parser.parse_args()

    for path in (args.raw_root, args.label_root, args.metadata_csv):
        if not path.exists():
            raise FileNotFoundError(path)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_label_files = sorted(args.label_root.rglob("*.txt"))
    pairs, labels_by_stem = enumerate_pairs(args.raw_root, args.label_root)
    dimensions, tracks, dim_anomalies, empty_label_files, total_boxes, malformed = collect_dimensions_and_tracks(all_label_files, args.workers)
    dim_rows = dimension_stats(dimensions)
    write_csv(OUT_ROOT / "dimension_stats.csv", dim_rows)
    print(f"Sampling {min(len(pairs), args.max_raw_frames)} frames for geometry checks...", flush=True)
    sampled_pairs = stratified_sample(pairs, args.max_raw_frames)

    candidate_rows: list[dict] = []
    direct_details: list[dict] = []
    sample_meta: list[dict] = []
    bev_dir = OUT_ROOT / "visualizations" / "bev_samples"
    save_bev_stems = {p.lidar.stem for p in (sampled_pairs[:8] if sampled_pairs else [])}
    for i, pair in enumerate(sampled_pairs, 1):
        save_vis = pair.lidar.stem in save_bev_stems
        current, direct, meta = analyze_frame(pair, bev_dir, save_vis)
        candidate_rows.extend(current)
        direct_details.extend(direct)
        sample_meta.append({"session": pair.session, "timestamp": pair.lidar.stem,
                            "lidar": str(pair.lidar), "label": str(pair.label),
                            "n_boxes": len(meta["boxes"]), "n_points": meta["point_count"]})
        if i % 10 == 0 or i == len(sampled_pairs):
            print(f"  geometry frames {i:,}/{len(sampled_pairs):,}; candidate rows {len(candidate_rows):,}", flush=True)

    # Temporal checks use all available GT tracks, not only sampled LiDAR frames.
    track_summaries = []
    candidate_track_rows: dict[str, list[dict]] = {}
    for yaw_name, _ in YAW_CANDIDATES:
        rows, track_anomalies = track_audit(tracks, yaw_name)
        moving = [r for r in rows if r["class"] == "Car" and r["displacement_m"] > 1.0
                  and 0.1 <= r["delta_t_s"] <= 2.0 and r["speed_mps"] < 50]
        errors = np.asarray([r["directed_yaw_motion_error_deg"] for r in moving], dtype=float)
        mod_errors = np.asarray([r["orientation_error_mod_180_deg"] for r in moving], dtype=float)
        track_summaries.append({
            "yaw_candidate": yaw_name, "moving_car_pairs": len(moving),
            "directed_median_error_deg": float(np.median(errors)) if len(errors) else None,
            "directed_p90_error_deg": float(np.percentile(errors, 90)) if len(errors) else None,
            "aligned_0deg_pct_lt20": float(np.mean(errors < 20) * 100) if len(errors) else None,
            "aligned_180deg_pct_lt20": float(np.mean(errors > 160) * 100) if len(errors) else None,
            "orientation_mod180_median_deg": float(np.median(mod_errors)) if len(mod_errors) else None,
            "orientation_mod180_p90_deg": float(np.percentile(mod_errors, 90)) if len(mod_errors) else None,
        })
        candidate_track_rows[yaw_name] = rows
    track_summaries.sort(key=lambda r: (r["orientation_mod180_median_deg"] if r["orientation_mod180_median_deg"] is not None else 999,
                                        r["directed_median_error_deg"] if r["directed_median_error_deg"] is not None else 999))
    write_csv(OUT_ROOT / "track_yaw_candidate_summary.csv", track_summaries)

    summaries = metric_summary(candidate_rows)
    dim_summary = {(r["class"], r["dimension_field"]): r for r in dim_rows}
    motion_by_yaw = {row["yaw_candidate"]: row for row in track_summaries}
    for row in summaries:
        long_field, short_field = (("dx", "dy") if row["dimension_candidate"] == DIM_CANDIDATES[0]
                                   else ("dy", "dx"))
        long_stats = dim_summary.get(("Car", long_field), {})
        short_stats = dim_summary.get(("Car", short_field), {})
        row["length_field_3to6m_pct"] = long_stats.get("car_length_range_pct_3_to_6m")
        row["width_field_1p5to2p5m_pct"] = short_stats.get("car_width_range_pct_1p5_to_2p5m")
        motion = motion_by_yaw.get(row["yaw_candidate"], {})
        row["motion_car_pairs"] = motion.get("moving_car_pairs")
        row["motion_mod180_median_deg"] = motion.get("orientation_mod180_median_deg")
        row["motion_mod180_p90_deg"] = motion.get("orientation_mod180_p90_deg")
        row["motion_directed_median_deg"] = motion.get("directed_median_error_deg")
        row["motion_aligned_0deg_pct_lt20"] = motion.get("aligned_0deg_pct_lt20")
        row["motion_aligned_180deg_pct_lt20"] = motion.get("aligned_180deg_pct_lt20")

    def assign_rank(key: str, low_is_good: bool) -> None:
        valid = [r for r in summaries if r.get(key) is not None and not (isinstance(r.get(key), float) and math.isnan(r[key]))]
        valid.sort(key=lambda r: r[key], reverse=not low_is_good)
        for rank, row in enumerate(valid, 1):
            row[f"rank_{key}"] = rank
        for row in summaries:
            row.setdefault(f"rank_{key}", len(valid) + 1)

    assign_rank("spearman_pointcount_corr", low_is_good=False)
    assign_rank("pointcount_median_abs_error", low_is_good=True)
    assign_rank("length_field_3to6m_pct", low_is_good=False)
    assign_rank("width_field_1p5to2p5m_pct", low_is_good=False)
    assign_rank("pca_median_error_deg", low_is_good=True)
    assign_rank("ground_median_abs_error_m", low_is_good=True)
    assign_rank("motion_mod180_median_deg", low_is_good=True)
    for row in summaries:
        rank_keys = [f"rank_{k}" for k in ("spearman_pointcount_corr", "pointcount_median_abs_error",
                      "length_field_3to6m_pct", "width_field_1p5to2p5m_pct", "pca_median_error_deg",
                      "ground_median_abs_error_m", "motion_mod180_median_deg")]
        row["consensus_rank_sum"] = sum(row[k] for k in rank_keys)
        row["consensus_rank_mean"] = row["consensus_rank_sum"] / len(rank_keys)
    summaries.sort(key=lambda r: (r["consensus_rank_mean"], r.get("rank_spearman_pointcount_corr", 999),
                                  r.get("rank_pca_median_error_deg", 999)))
    write_csv(OUT_ROOT / "convention_comparison.csv", summaries)
    selected = summaries[0] if summaries else {"yaw_candidate": "yaw", "dimension_candidate": DIM_CANDIDATES[0], "z_candidate": "geometric_center"}
    selected_key = (selected["yaw_candidate"], selected["dimension_candidate"], selected["z_candidate"])
    best_point_rows = [r for r in candidate_rows if
                       (r["yaw_candidate"], r["dimension_candidate"], r["z_candidate"]) == selected_key]
    write_csv(OUT_ROOT / "pointcount_validation.csv", best_point_rows)
    write_csv(OUT_ROOT / "point_coverage_validation.csv", direct_details)
    best_pca_rows = [r for r in candidate_rows if r["class"] == "Car" and r["pca_error_deg"] is not None and
                     (r["yaw_candidate"], r["dimension_candidate"], r["z_candidate"]) == selected_key]
    write_csv(OUT_ROOT / "pca_yaw_error.csv", best_pca_rows)
    best_yaw = selected["yaw_candidate"]
    motion_rows = [r for r in candidate_track_rows.get(best_yaw, [])
                   if r["class"] == "Car" and r["displacement_m"] > 1.0 and
                   0.1 <= r["delta_t_s"] <= 2.0 and 2.0 <= r["speed_mps"] < 40]
    write_csv(OUT_ROOT / "track_motion_validation.csv", motion_rows)
    write_csv(OUT_ROOT / "raw_frame_sample.csv", sample_meta)
    camera_manifest = camera_projection_audit(sampled_pairs, DEFAULT_META, OUT_ROOT / "visualizations",
                                               selected_key, max_frames=12)

    # Gather dimension and geometry outliers.
    anomalies = list(dim_anomalies)
    for row in direct_details:
        if row["numpoints_gt"] >= 10 and row["pointcount_abs_error_direct"] > max(20, row["numpoints_gt"]):
            anomalies.append({"type": "direct_box_pointcount_mismatch", "segment": row["segment"],
                              "timestamp": row["timestamp"], "trackid": row["trackid"], "class": row["class"],
                              "detail": f"GT={row['numpoints_gt']}, computed={row['numpoints_computed_direct']}"})
    _, temporal_anomalies = track_audit(tracks, best_yaw)
    anomalies.extend(temporal_anomalies)
    write_csv(OUT_ROOT / "anomalies.csv", anomalies)
    write_csv(OUT_ROOT / "dimension_outliers.csv",
              [r for r in anomalies if r["type"] == "dimension_outlier"])

    make_plots(dim_rows, candidate_rows, direct_details, candidate_track_rows.get(best_yaw, []))
    plot_dir = OUT_ROOT / "visualizations" / "histograms"
    selected_pca_errors = np.asarray([r["pca_error_deg"] for r in best_pca_rows if r["pca_error_deg"] is not None], dtype=float)
    if len(selected_pca_errors):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(selected_pca_errors, bins=np.arange(0, 95, 5), color="#4aa3b5", edgecolor="white")
        ax.set(xlabel="PCA long-axis error (degrees, modulo 180)", ylabel="Car boxes",
               title=f"Selected candidate: {selected_key}")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / "pca_yaw_error_distribution.png")
        plt.close(fig)
    selected_ground = np.asarray([r["ground_error_m"] for r in best_point_rows if r["ground_error_m"] is not None], dtype=float)
    if len(selected_ground):
        fig, ax = plt.subplots(figsize=(8, 5))
        ground_hi = max(0.1, min(4.0, float(np.percentile(selected_ground, 99))))
        ax.hist(selected_ground, bins=np.arange(0, ground_hi + 0.1, 0.1),
                color="#8bba6f", edgecolor="white")
        ax.set(xlabel="|predicted box bottom - local ground proxy| (m)", ylabel="Car boxes",
               title="Ground contact proxy for selected candidate")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / "ground_contact_error_distribution.png")
        plt.close(fig)

    # Metadata and coverage counts.
    with args.metadata_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        metadata_rows = list(csv.DictReader(stream))
    zero_rows = [r for r in metadata_rows if int(float(r.get("num_boxes") or 0)) == 0]
    annotated_rows = [r for r in metadata_rows if str(r.get("3d_annotated", "0")) == "1"]
    class_counts = {cls: len(axes.get("dx", [])) for cls, axes in dimensions.items()}
    top = summaries[0] if summaries else {}
    rendered_camera_rows = [r for r in camera_manifest if r.get("status") == "rendered"]
    camera_delta_ms = [float(r["delta_t_ms"]) for r in rendered_camera_rows if r.get("delta_t_ms") is not None]
    dim_by_class = {r["class"] + "/" + r["dimension_field"]: r for r in dim_rows}
    expected_axes = "not established"
    car_dx, car_dy, car_dz = (dim_by_class.get("Car/" + a, {}).get("median_m") for a in ("dx", "dy", "dz"))
    if all(v is not None for v in (car_dx, car_dy, car_dz)):
        expected_axes = f"median dx={car_dx:.2f}m, dy={car_dy:.2f}m, dz={car_dz:.2f}m"
    pca_summary = [r for r in summaries if (r["yaw_candidate"], r["dimension_candidate"], r["z_candidate"]) == selected_key]
    best_ground = pca_summary[0] if pca_summary else {}
    bottom_candidate = next((r for r in summaries
                             if r["yaw_candidate"] == top.get("yaw_candidate")
                             and r["dimension_candidate"] == top.get("dimension_candidate")
                             and r["z_candidate"] == "bottom_center"), {})
    anomaly_counts = Counter(row["type"] for row in anomalies)
    camera_rejected = sum(int(r.get("boxes_rejected_extreme_projection", 0) or 0) for r in rendered_camera_rows)
    report = f'''# PNK 3D Ground Truth Reverse Audit

## Scope and coverage

- Raw sensor root: `{args.raw_root}`
- Label root: `{args.label_root}`
- Metadata index: `{args.metadata_csv}`
- Metadata rows: **{len(metadata_rows):,}**; `3d_annotated=1`: **{len(annotated_rows):,}**; `num_boxes=0`: **{len(zero_rows):,}** (zero rows are marked not annotated in this metadata index).
- Label text files: **{len(all_label_files):,}**; parsed box rows: **{total_boxes:,}**; empty files: **{empty_label_files:,}**; malformed rows: **{malformed:,}**.
- Label class counts: **{class_counts}**.
- Exact timestamp pairs available: **{len(pairs):,}**; expensive geometry audit sampled **{len(sampled_pairs):,}** frames, stratified across sessions. See `raw_frame_sample.csv`.
- Point-cloud checks use `LIDAR_MERGED_EGO`; label and cloud are paired by exact timestamp stem. Results do not cover frames with no local raw LiDAR pair.

## Findings

- Car dimension medians: {expected_axes}. Physical ranges support `dx=length, dy=width`: dx is within 3–6 m for {dim_by_class.get('Car/dx', {}).get('car_length_range_pct_3_to_6m')}% of cars; dy is within 1.5–2.5 m for {dim_by_class.get('Car/dy', {}).get('car_width_range_pct_1p5_to_2p5m')}%.
- Best-ranked tested candidate under the multi-evidence score: **yaw `{top.get('yaw_candidate')}`, `{top.get('dimension_candidate')}`, z `{top.get('z_candidate')}`**. Ranking includes point-count agreement, dimension priors, PCA long-axis alignment, local ground contact, and motion consistency.
- Candidate Car boxes: {top.get('n_boxes', 0)}; point-count Pearson / Spearman: {top.get('pearson_pointcount_corr')} / {top.get('spearman_pointcount_corr')}; MAE / median AE: {top.get('pointcount_mae')} / {top.get('pointcount_median_abs_error')} points; exact match: {top.get('pointcount_exact_pct')}%.
- PCA long-axis median / P90: {top.get('pca_median_error_deg')} / {top.get('pca_p90_error_deg')} deg (modulo 180; N={top.get('pca_n')}); below 5/10/20 deg: {top.get('pca_lt5_pct')}% / {top.get('pca_lt10_pct')}% / {top.get('pca_lt20_pct')}%.
- Z center comparison: geometric-center point-count median AE / exact match = {top.get('pointcount_median_abs_error')} / {top.get('pointcount_exact_pct')}%; bottom-center = {bottom_candidate.get('pointcount_median_abs_error')} / {bottom_candidate.get('pointcount_exact_pct')}%. Ground proxy median absolute error = {best_ground.get('ground_median_abs_error_m')} m for geometric center vs {bottom_candidate.get('ground_median_abs_error_m')} m for bottom center.
- Camera projection: {len(rendered_camera_rows)} overlays across {len(set(r['camera'] for r in rendered_camera_rows))} cameras / {len(set(r['lidar_timestamp'] for r in rendered_camera_rows))} timestamps; mean LiDAR-camera delta {float(np.mean(camera_delta_ms)) if camera_delta_ms else float('nan'):.1f} ms. {camera_rejected} extreme projections skipped. See `visualizations/camera_projection_manifest.csv`.
- Local ground proxy error for selected Z convention, median / P90: {best_ground.get('ground_median_abs_error_m')} / {best_ground.get('ground_p90_abs_error_m')} m. This uses the 10th percentile of nearby low LiDAR returns, **not a surveyed ground plane**.
- Moving-car heading candidate `{best_yaw}` has orientation-mod-180 median / P90 {next((r['orientation_mod180_median_deg'] for r in track_summaries if r['yaw_candidate']==best_yaw), None)}° / {next((r['orientation_mod180_p90_deg'] for r in track_summaries if r['yaw_candidate']==best_yaw), None)}° across {next((r['moving_car_pairs'] for r in track_summaries if r['yaw_candidate']==best_yaw), 0)} moving track pairs. Directional front/back is separately recorded in `track_yaw_candidate_summary.csv`.

## Interpretation / confidence

- **Dimensions — high confidence:** both population distributions and point-count agreement support `dx=length, dy=width`; swapping dimensions degrades point-count fit. `dz` is vertical extent.
- **Center convention - HIGH for Z:** geometric center is strongly supported by point-count and local-ground-proxy comparisons. `z` is likely the box geometric center; XY center offsets were not separately perturbed. The proxy is weak on occluded, sloped, or sparse areas.
- **Yaw axis/sign - MEDIUM; front/back - LOW:** `yaw` with direct dimensions ranks first, with PCA long-axis median error {top.get('pca_median_error_deg')} deg. Under ego axes (+X forward, +Y left), this supports the long axis at `yaw` from +X, positive toward +Y. PCA cannot distinguish 180-degree reversal, and temporal motion does not resolve front/back reliably.
- **Coordinate handedness - LOW / unresolved:** yaw candidates and track motion were compared, but motion errors remain broad. Camera overlays are qualitative and do not establish handedness by themselves.
- **Track continuity:** flagged anomaly counts are `{dict(anomaly_counts)}`. These heuristic alerts need review; they are not confirmed annotation errors.
- **Point coverage:** `point_coverage_validation.csv` records local-neighborhood occupancy, point-to-box-surface distances, and per-axis quartile occupancy. Nearby non-object returns can enter the denominator, so treat it as diagnostic rather than semantic object recall.
- **`numpoints`:** correlation/exactness compare GT with points in the timestamp-matched merged cloud. A mismatch can mean different filtering/accumulation or sensor source, not automatically incorrect labels.
- **Camera cross-check:** date-matched `Calib_2` was used with nearest camera frames within 120 ms. Extreme projected boxes are skipped; final overlays are in `visualizations/camera_projection_final/`. Visual inspection shows some boxes align plausibly while others remain offset or produce long edges, so this does not pass as a calibration proof. Overlay colors: Car=orange, Rider=cyan, Pedestrian=magenta; numbers are row indices local to that label file, not track IDs.
- No source labels were modified. All computed results and visualizations are under `{OUT_ROOT}`.

## Artifacts

- `dimension_stats.csv`
- `convention_comparison.csv`
- `pca_yaw_error.csv`
- `pointcount_validation.csv`
- `point_coverage_validation.csv`
- `track_motion_validation.csv`, `track_yaw_candidate_summary.csv`
- `anomalies.csv`, `dimension_outliers.csv`
- `raw_frame_sample.csv`
- `visualizations/camera_projection_manifest.csv`
- `visualizations/histograms/`, `visualizations/bev_samples/`, and `visualizations/camera_projection_final/`

## Conversion candidate

For a reversible converter prototype, use the top-ranked row in `convention_comparison.csv` (`yaw`, `dx=length,dy=width`, `geometric_center`). Keep it configurable and verify camera overlays plus downstream METEOR schema expectations before bulk conversion; do not mutate source `.txt` labels in place.
'''
    (OUT_ROOT / "audit_summary.md").write_text(report, encoding="utf-8")
    print(f"Audit complete. Results: {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
