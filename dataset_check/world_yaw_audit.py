"""Read-only audit of PNK box yaw after ego-pose compensation into local ENU.

All derived files are written under dataset_check/world_yaw_audit. Raw sensor,
NAV and label files are only read.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_pnk_gt as base


OUT = Path(__file__).resolve().parent / "world_yaw_audit"
RAW = base.DEFAULT_RAW
LABELS = base.DEFAULT_LABELS
META = base.DEFAULT_META
NAV_FIELDS = ("Heading", "Heading2")
YAW_CANDIDATES = base.YAW_CANDIDATES
MAX_NAV_GAP_S = 0.10
MAX_EGO_TURN_DEG_S = 8.0
MIN_OBJECT_SPEED_MPS = 1.5
MAX_OBJECT_SPEED_MPS = 40.0
MAX_AXIS_MOTION_ERROR_DEG = 25.0
MAX_PROPAGATION_GAP_S = 0.75


def parse_ts(text: str) -> float:
    sec, frac = text.split("-", 1)
    return int(sec) + int(frac) / 1_000_000_000.0


def wrap2pi(x: float) -> float:
    return x % (2.0 * math.pi)


def wrap_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def angle_error(a: float, b: float) -> float:
    return abs(wrap_pi(a - b))


def angle_error_mod_pi(a: float, b: float) -> float:
    return abs((a - b + math.pi / 2.0) % math.pi - math.pi / 2.0)


def circular_interp_deg(a: float, b: float, w: float) -> float:
    ar, br = math.radians(a), math.radians(b)
    return math.degrees(wrap2pi(ar + w * wrap_pi(br - ar))) % 360.0


def safe_float(value: str | None, default: float = float("nan")) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


@dataclass
class NavSeries:
    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    heading: np.ndarray
    heading2: np.ndarray
    ve: np.ndarray
    vn: np.ndarray
    speed: np.ndarray
    heading_std: np.ndarray
    heading2_std: np.ndarray
    status: list[str]
    age: list[str]
    warning: list[str]
    origin_lat: float
    origin_lon: float
    origin_alt: float
    origin_ecef: np.ndarray
    origin_lon_rad: float
    origin_lat_rad: float

    def interpolate(self, timestamp: float) -> dict | None:
        idx = int(np.searchsorted(self.t, timestamp, side="left"))
        if idx < len(self.t) and abs(float(self.t[idx]) - timestamp) < 1e-6:
            lo = hi = idx
            weight = 0.0
        elif idx == 0 or idx >= len(self.t):
            return None
        else:
            lo, hi = idx - 1, idx
            gap = float(self.t[hi] - self.t[lo])
            if gap <= 0 or gap > MAX_NAV_GAP_S:
                return None
            weight = (timestamp - float(self.t[lo])) / gap
        nearest_dt = min(abs(timestamp - float(self.t[lo])), abs(timestamp - float(self.t[hi])))

        def interp(arr: np.ndarray) -> float:
            a, b = float(arr[lo]), float(arr[hi])
            if not (math.isfinite(a) and math.isfinite(b)):
                return float("nan")
            return a + weight * (b - a)

        def interp_heading(arr: np.ndarray) -> float:
            a, b = float(arr[lo]), float(arr[hi])
            if not (math.isfinite(a) and math.isfinite(b)):
                return float("nan")
            return circular_interp_deg(a, b, weight)

        heading = interp_heading(self.heading)
        heading2 = interp_heading(self.heading2)
        lat, lon, alt = interp(self.lat), interp(self.lon), interp(self.alt)
        if not all(math.isfinite(x) for x in (lat, lon, alt)):
            return None
        return {
            "timestamp_s": timestamp,
            "nearest_nav_delta_ms": nearest_dt * 1000.0,
            "nav_bracket_gap_ms": (float(self.t[hi] - self.t[lo]) * 1000.0 if hi != lo else 0.0),
            "lat": lat, "lon": lon, "alt": alt,
            "Heading": heading, "Heading2": heading2,
            "Ve": interp(self.ve), "Vn": interp(self.vn), "V_2D": interp(self.speed),
            "Heading_std": interp(self.heading_std), "Heading2_std": interp(self.heading2_std),
            "Status": self.status[lo] if weight <= 0.5 else self.status[hi],
            "Age": self.age[lo] if weight <= 0.5 else self.age[hi],
            "Warning": self.warning[lo] if weight <= 0.5 else self.warning[hi],
        }

    def to_enu(self, lat: float, lon: float, alt: float) -> tuple[float, float, float]:
        a = 6378137.0
        e2 = 6.69437999014e-3

        def ecef(la_deg: float, lo_deg: float, h: float) -> np.ndarray:
            la, lo = math.radians(la_deg), math.radians(lo_deg)
            n = a / math.sqrt(1.0 - e2 * math.sin(la) ** 2)
            return np.array([(n + h) * math.cos(la) * math.cos(lo),
                             (n + h) * math.cos(la) * math.sin(lo),
                             (n * (1.0 - e2) + h) * math.sin(la)], dtype=np.float64)

        delta = ecef(lat, lon, alt) - self.origin_ecef
        slat, clat = math.sin(self.origin_lat_rad), math.cos(self.origin_lat_rad)
        slon, clon = math.sin(self.origin_lon_rad), math.cos(self.origin_lon_rad)
        east = -slon * delta[0] + clon * delta[1]
        north = -slat * clon * delta[0] - slat * slon * delta[1] + clat * delta[2]
        up = clat * clon * delta[0] + clat * slon * delta[1] + slat * delta[2]
        return float(east), float(north), float(up)


def load_nav(session_dir: Path, target_times: list[float]) -> NavSeries:
    records = []
    nav_dir = session_dir / "OTHERS" / "NAV"
    low_t, high_t = min(target_times), max(target_times)
    for path in sorted(nav_dir.glob("*.csv")):
        try:
            chunk_start = parse_ts(path.stem)
        except ValueError:
            continue
        # NAV files are time chunks. Read only chunks around this session's
        # labeled frames, with one chunk of margin for interpolation endpoints.
        if chunk_start > high_t + 0.1 or chunk_start + 61.0 < low_t - 0.1:
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    timestamp = parse_ts(row["Timestamp"])
                    lat, lon, alt = (safe_float(row.get(k)) for k in ("Latitude", "Longitude", "Altitude"))
                    if not all(math.isfinite(x) for x in (lat, lon, alt)):
                        continue
                    records.append({
                        "t": timestamp, "lat": lat, "lon": lon, "alt": alt,
                        "Heading": safe_float(row.get("Heading")),
                        "Heading2": safe_float(row.get("Heading2")),
                        "Ve": safe_float(row.get("Ve")), "Vn": safe_float(row.get("Vn")),
                        "V_2D": safe_float(row.get("V_2D")),
                        "Heading_std": safe_float(row.get("Heading_std")),
                        "Heading2_std": safe_float(row.get("Heading2_std")),
                        "Status": row.get("Status", ""), "Age": row.get("Age", ""),
                        "Warning": row.get("Warning", ""),
                    })
                except (KeyError, ValueError, OverflowError):
                    continue
    records.sort(key=lambda r: r["t"])
    # Some NAV file chunks can share a boundary row. Keep the first copy.
    unique = []
    for row in records:
        if not unique or row["t"] > unique[-1]["t"]:
            unique.append(row)
    if len(unique) < 2:
        raise RuntimeError(f"Insufficient NAV rows in {nav_dir}")
    origin = unique[0]
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat0, lon0 = math.radians(origin["lat"]), math.radians(origin["lon"])
    n0 = a / math.sqrt(1.0 - e2 * math.sin(lat0) ** 2)
    ecef0 = np.array([(n0 + origin["alt"]) * math.cos(lat0) * math.cos(lon0),
                      (n0 + origin["alt"]) * math.cos(lat0) * math.sin(lon0),
                      (n0 * (1.0 - e2) + origin["alt"]) * math.sin(lat0)])
    def arr(name: str) -> np.ndarray:
        return np.asarray([r[name] for r in unique], dtype=np.float64)
    return NavSeries(
        t=arr("t"), lat=arr("lat"), lon=arr("lon"), alt=arr("alt"),
        heading=arr("Heading"), heading2=arr("Heading2"), ve=arr("Ve"), vn=arr("Vn"),
        speed=arr("V_2D"), heading_std=arr("Heading_std"), heading2_std=arr("Heading2_std"),
        status=[r["Status"] for r in unique], age=[r["Age"] for r in unique],
        warning=[r["Warning"] for r in unique], origin_lat=origin["lat"], origin_lon=origin["lon"],
        origin_alt=origin["alt"], origin_ecef=ecef0, origin_lon_rad=lon0, origin_lat_rad=lat0)


def nav_course_metrics(nav: NavSeries, field: str) -> dict:
    heading = nav.heading if field == "Heading" else nav.heading2
    mask = np.isfinite(heading) & np.isfinite(nav.ve) & np.isfinite(nav.vn) & (nav.speed >= 2.0)
    if not np.any(mask):
        return {"heading_field": field, "n_speed_ge_2mps": 0}
    course = np.degrees(np.arctan2(nav.ve[mask], nav.vn[mask])) % 360.0
    error = np.abs((course - heading[mask] + 180.0) % 360.0 - 180.0)
    return {"heading_field": field, "n_speed_ge_2mps": int(len(error)),
            "course_error_median_deg": float(np.median(error)),
            "course_error_p90_deg": float(np.percentile(error, 90)),
            "course_error_lt_5deg_pct": float(np.mean(error < 5.0) * 100),
            "course_error_lt_15deg_pct": float(np.mean(error < 15.0) * 100)}


def add_world_pose(obs: dict, nav: NavSeries, pose: dict, heading_field: str) -> dict | None:
    compass = pose.get(heading_field, float("nan"))
    heading_std = pose.get("Heading_std" if heading_field == "Heading" else "Heading2_std", float("nan"))
    # Keep Heading and Heading2 comparisons on identical observations. Requiring
    # both quality fields prevents a larger, lower-quality sample for one field.
    heading_std_common = pose.get("Heading_std", float("nan"))
    heading2_std_common = pose.get("Heading2_std", float("nan"))
    heading_common = pose.get("Heading", float("nan"))
    heading2_common = pose.get("Heading2", float("nan"))
    if (not math.isfinite(compass)
            or not math.isfinite(heading_common)
            or not math.isfinite(heading2_common)
            or not math.isfinite(heading_std_common)
            or not math.isfinite(heading2_std_common)
            or heading_std_common > 2.0
            or heading2_std_common > 2.0):
        return None
    # NAV heading is tested against Ve/Vn as compass degrees clockwise from North.
    # Ego +X is forward and +Y is left; local ENU yaw is therefore 90deg - compass.
    ego_yaw = math.radians(90.0 - compass)
    origin = pose["enu"]
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    xw = origin[0] + c * obs["x"] - s * obs["y"]
    yw = origin[1] + s * obs["x"] + c * obs["y"]
    return {**obs, "heading_field": heading_field, "ego_yaw_world_rad": wrap2pi(ego_yaw),
            "world_x_m": xw, "world_y_m": yw, "nav_heading_deg": compass,
            "nav_heading_std_deg": heading_std,
            "nav_heading_std_common_deg": heading_std_common,
            "nav_heading2_std_common_deg": heading2_std_common,
            "nav_nearest_delta_ms": pose["nearest_nav_delta_ms"],
            "nav_bracket_gap_ms": pose["nav_bracket_gap_ms"], "nav_status": pose["Status"],
            "nav_age": pose["Age"], "nav_warning": pose["Warning"],
            "ego_speed_mps": pose["V_2D"]}


def estimate_motion(track: list[dict]) -> None:
    track.sort(key=lambda r: r["timestamp_s"])
    n = len(track)
    for i, row in enumerate(track):
        row["world_vx_mps"] = float("nan")
        row["world_vy_mps"] = float("nan")
        row["world_speed_mps"] = float("nan")
        row["world_motion_yaw_rad"] = float("nan")
        row["ego_turn_rate_deg_s"] = float("nan")
        if n < 2:
            continue
        left, right = max(0, i - 1), min(n - 1, i + 1)
        if left == right:
            continue
        dt = track[right]["timestamp_s"] - track[left]["timestamp_s"]
        if dt <= 0 or dt > 1.0:
            continue
        vx = (track[right]["world_x_m"] - track[left]["world_x_m"]) / dt
        vy = (track[right]["world_y_m"] - track[left]["world_y_m"]) / dt
        speed = math.hypot(vx, vy)
        row["world_vx_mps"], row["world_vy_mps"] = vx, vy
        row["world_speed_mps"] = speed
        row["world_motion_yaw_rad"] = math.atan2(vy, vx) if speed > 0 else float("nan")

        nav_left, nav_right = track[left]["ego_yaw_world_rad"], track[right]["ego_yaw_world_rad"]
        row["ego_turn_rate_deg_s"] = math.degrees(wrap_pi(nav_right - nav_left)) / dt


def moving_eligible_keys(track_groups: dict) -> set[tuple[str, str, str]]:
    eligible = set()
    for track in track_groups.values():
        for row in track:
            speed = row.get("world_speed_mps", float("nan"))
            motion = row.get("world_motion_yaw_rad", float("nan"))
            turn = row.get("ego_turn_rate_deg_s", float("nan"))
            heading_std = row.get("nav_heading_std_deg", float("nan"))
            if not (math.isfinite(speed) and math.isfinite(motion)):
                continue
            if not (MIN_OBJECT_SPEED_MPS <= speed <= MAX_OBJECT_SPEED_MPS):
                continue
            if not math.isfinite(turn) or abs(turn) > MAX_EGO_TURN_DEG_S:
                continue
            if not math.isfinite(heading_std) or heading_std > 2.0:
                continue
            eligible.add((row["segment"], str(row["trackid"]), str(row["timestamp"])))
    return eligible


def moving_metrics(track_groups: dict, candidate_name: str,
                   eligible_keys: set[tuple[str, str, str]] | None = None) -> dict:
    yaw_fn = dict(YAW_CANDIDATES)[candidate_name]
    errors, speeds, directed_errors = [], [], []
    considered = 0
    for track in track_groups.values():
        for row in track:
            if eligible_keys is not None and (row["segment"], str(row["trackid"]),
                                              str(row["timestamp"])) not in eligible_keys:
                continue
            speed = row.get("world_speed_mps", float("nan"))
            motion = row.get("world_motion_yaw_rad", float("nan"))
            if not (math.isfinite(speed) and math.isfinite(motion)):
                continue
            if not (MIN_OBJECT_SPEED_MPS <= speed <= MAX_OBJECT_SPEED_MPS):
                continue
            if not math.isfinite(row.get("ego_turn_rate_deg_s", float("nan"))) or abs(row["ego_turn_rate_deg_s"]) > MAX_EGO_TURN_DEG_S:
                continue
            if not math.isfinite(row.get("nav_heading_std_deg", float("nan"))) or row["nav_heading_std_deg"] > 2.0:
                continue
            axis = wrap2pi(row["ego_yaw_world_rad"] + yaw_fn(row["yaw"]))
            error = math.degrees(angle_error_mod_pi(axis, motion))
            errors.append(error)
            speeds.append(speed)
            directed_errors.append(math.degrees(min(angle_error(axis, motion), angle_error(axis + math.pi, motion))))
            considered += 1
    if not errors:
        return {"moving_observations": 0}
    arr = np.asarray(errors)
    return {"moving_observations": considered, "moving_speed_median_mps": float(np.median(speeds)),
            "motion_mod180_median_deg": float(np.median(arr)), "motion_mod180_p90_deg": float(np.percentile(arr, 90)),
            "motion_mod180_lt10_pct": float(np.mean(arr < 10) * 100),
            "motion_mod180_lt20_pct": float(np.mean(arr < 20) * 100),
            "motion_mod180_lt25_pct": float(np.mean(arr <= MAX_AXIS_MOTION_ERROR_DEG) * 100),
            "directed_best_error_median_deg": float(np.median(directed_errors))}


def static_track_jitter(track_groups: dict, candidate_name: str) -> dict:
    yaw_fn = dict(YAW_CANDIDATES)[candidate_name]
    track_medians, point_errors = [], []
    used_tracks = 0
    for track in track_groups.values():
        eligible = [r for r in track if math.isfinite(r.get("world_speed_mps", float("nan")))
                    and r["world_speed_mps"] <= 0.75]
        if len(eligible) < 4 or eligible[-1]["timestamp_s"] - eligible[0]["timestamp_s"] < 0.6:
            continue
        ego_yaws = np.unwrap(np.asarray([r["ego_yaw_world_rad"] for r in eligible]))
        if math.degrees(float(np.ptp(ego_yaws))) < 8.0:
            continue
        angles = np.asarray([wrap2pi(r["ego_yaw_world_rad"] + yaw_fn(r["yaw"])) for r in eligible])
        mean = 0.5 * math.atan2(float(np.mean(np.sin(2 * angles))), float(np.mean(np.cos(2 * angles))))
        err = np.degrees([angle_error_mod_pi(a, mean) for a in angles])
        track_medians.append(float(np.median(err)))
        point_errors.extend(err.tolist())
        used_tracks += 1
    if not track_medians:
        return {"static_tracks_with_ego_turn": 0}
    return {"static_tracks_with_ego_turn": used_tracks,
            "static_axis_jitter_track_median_deg": float(np.median(track_medians)),
            "static_axis_jitter_p90_deg": float(np.percentile(point_errors, 90))}


def candidate_table(track_groups_by_field: dict) -> list[dict]:
    rows = []
    # World positions and thus derived speeds/turn rates vary slightly by NAV
    # heading field. Rank yaw conventions on only observations passing the
    # kinematic gates under BOTH fields.
    eligible_by_field = {field: moving_eligible_keys(groups)
                         for field, groups in track_groups_by_field.items()}
    common_eligible = set.intersection(*(eligible_by_field[f] for f in NAV_FIELDS))
    for field, groups in track_groups_by_field.items():
        for name, _ in YAW_CANDIDATES:
            motion = moving_metrics(groups, name, common_eligible)
            static = static_track_jitter(groups, name)
            rows.append({"nav_heading_field": field, "yaw_candidate": name,
                         "common_moving_sample": len(common_eligible), **motion, **static})
    rows.sort(key=lambda r: (r.get("motion_mod180_median_deg", 999),
                             r.get("static_axis_jitter_track_median_deg", 999),
                             -r.get("moving_observations", 0)))
    for idx, row in enumerate(rows, 1):
        row["motion_rank"] = idx
    # Rank separately within each NAV heading field.
    for field in NAV_FIELDS:
        selected = [r for r in rows if r["nav_heading_field"] == field]
        selected.sort(key=lambda r: (r.get("motion_mod180_median_deg", 999),
                                     r.get("static_axis_jitter_track_median_deg", 999),
                                     -r.get("moving_observations", 0)))
        for rank, row in enumerate(selected, 1):
            row["rank_within_heading_field"] = rank
    return rows


def choose_config(rows: list[dict]) -> dict:
    viable = [r for r in rows if r.get("moving_observations", 0) >= 50]
    if not viable:
        return rows[0]
    return min(viable, key=lambda r: (r.get("motion_mod180_median_deg", 999),
                                      r.get("static_axis_jitter_track_median_deg", 999),
                                      -r.get("moving_observations", 0)))


def resolve_track_headings(track_groups: dict, config: dict) -> list[dict]:
    yaw_fn = dict(YAW_CANDIDATES)[config["yaw_candidate"]]
    output = []
    for track_key, track in track_groups.items():
        track.sort(key=lambda r: r["timestamp_s"])
        n = len(track)
        axes = [wrap2pi(r["ego_yaw_world_rad"] + yaw_fn(r["yaw"])) for r in track]
        resolved: list[float | None] = [None] * n
        sources = ["heading_invalid"] * n
        confidence = [0.0] * n
        anchor_indices = []
        for i, row in enumerate(track):
            speed = row.get("world_speed_mps", float("nan"))
            motion = row.get("world_motion_yaw_rad", float("nan"))
            turn = row.get("ego_turn_rate_deg_s", float("nan"))
            if not (math.isfinite(speed) and math.isfinite(motion) and math.isfinite(turn)):
                continue
            if not (MIN_OBJECT_SPEED_MPS <= speed <= MAX_OBJECT_SPEED_MPS) or abs(turn) > MAX_EGO_TURN_DEG_S:
                continue
            if not math.isfinite(row.get("nav_heading_std_deg", float("nan"))) or row["nav_heading_std_deg"] > 2.0:
                continue
            error = math.degrees(angle_error_mod_pi(axes[i], motion))
            if error > MAX_AXIS_MOTION_ERROR_DEG:
                continue
            forward_error, backward_error = angle_error(axes[i], motion), angle_error(axes[i] + math.pi, motion)
            resolved[i] = axes[i] if forward_error <= backward_error else wrap2pi(axes[i] + math.pi)
            sources[i] = "world_motion_assumed_forward"
            confidence[i] = max(0.0, 1.0 - error / MAX_AXIS_MOTION_ERROR_DEG) * min(1.0, speed / 5.0)
            anchor_indices.append(i)

        # Propagate the front/back branch through contiguous portions of the
        # same track. The modulo-pi axis is still taken from each frame.
        chunks = []
        start = 0
        for i in range(1, n):
            if track[i]["timestamp_s"] - track[i - 1]["timestamp_s"] > MAX_PROPAGATION_GAP_S:
                chunks.append((start, i))
                start = i
        chunks.append((start, n))
        for begin, end in chunks:
            anchors = [i for i in anchor_indices if begin <= i < end]
            if not anchors:
                continue
            first, last = anchors[0], anchors[-1]
            # Fill before the first motion anchor, walking backwards.
            for i in range(first - 1, begin - 1, -1):
                if resolved[i + 1] is None:
                    break
                opts = (axes[i], wrap2pi(axes[i] + math.pi))
                resolved[i] = min(opts, key=lambda a: angle_error(a, resolved[i + 1]))
                sources[i] = "track_propagation"
                confidence[i] = confidence[first] * math.exp(-(track[first]["timestamp_s"] - track[i]["timestamp_s"]) / 3.0)
            # Fill forward; a new motion anchor resets the propagated branch.
            for i in range(first + 1, end):
                if resolved[i] is not None and sources[i] == "world_motion_assumed_forward":
                    continue
                if resolved[i - 1] is None:
                    continue
                opts = (axes[i], wrap2pi(axes[i] + math.pi))
                resolved[i] = min(opts, key=lambda a: angle_error(a, resolved[i - 1]))
                sources[i] = "track_propagation"
                nearest_anchor = min(anchors, key=lambda j: abs(track[j]["timestamp_s"] - track[i]["timestamp_s"]))
                confidence[i] = confidence[nearest_anchor] * math.exp(-abs(track[nearest_anchor]["timestamp_s"] - track[i]["timestamp_s"]) / 3.0)

        for i, row in enumerate(track):
            output.append({
                "segment": row["segment"], "timestamp": row["timestamp"], "timestamp_s": row["timestamp_s"],
                "trackid": row["trackid"], "class": row["class"], "x_ego_m": row["x"], "y_ego_m": row["y"],
                "yaw_raw_rad": row["yaw"], "yaw_candidate": config["yaw_candidate"],
                "nav_heading_field": config["nav_heading_field"], "nav_heading_deg": row["nav_heading_deg"],
                "world_x_enu_m": row["world_x_m"], "world_y_enu_m": row["world_y_m"],
                "world_axis_yaw_mod_pi_deg": math.degrees(axes[i] % math.pi),
                "world_speed_mps": row.get("world_speed_mps"),
                "motion_heading_world_deg": (math.degrees(wrap2pi(row["world_motion_yaw_rad"]))
                                              if math.isfinite(row.get("world_motion_yaw_rad", float("nan"))) else None),
                "motion_axis_error_mod180_deg": (math.degrees(angle_error_mod_pi(axes[i], row["world_motion_yaw_rad"]))
                                                 if math.isfinite(row.get("world_motion_yaw_rad", float("nan"))) else None),
                "ego_turn_rate_deg_s": row.get("ego_turn_rate_deg_s"),
                "nav_nearest_delta_ms": row["nav_nearest_delta_ms"],
                "heading_world_deg": math.degrees(resolved[i]) % 360.0 if resolved[i] is not None else None,
                "heading_valid": resolved[i] is not None,
                "heading_source": sources[i], "heading_confidence": confidence[i],
            })
    return output


def inspect_point_time(pairs: list[base.Pair]) -> dict:
    try:
        import laspy
        path = pairs[0].lidar
        cloud = laspy.read(path)
        values = np.asarray(cloud.point_timestamp, dtype=np.float64) / 1e9
        frame_t = parse_ts(path.stem)
        offsets = values - frame_t
        return {"sample_laz": path.name, "point_timestamp_type": str(cloud.point_timestamp.dtype),
                "point_count": int(len(values)), "scan_offset_min_ms": float(np.min(offsets) * 1000),
                "scan_offset_median_ms": float(np.median(offsets) * 1000),
                "scan_offset_max_ms": float(np.max(offsets) * 1000)}
    except Exception as exc:
        return {"sample_laz": "unavailable", "point_timestamp_error": str(exc)}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs, _ = base.enumerate_pairs(RAW, LABELS)
    if not pairs:
        raise RuntimeError("No exact label/LIDAR_MERGED_EGO timestamp pairs found")

    session_dirs = {p.name: p for p in RAW.iterdir() if p.is_dir()}
    needed_sessions = sorted({pair.session for pair in pairs})
    times_by_session = defaultdict(list)
    for pair in pairs:
        times_by_session[pair.session].append(parse_ts(pair.lidar.stem))
    nav_by_session = {}
    nav_summaries = []
    nav_course = []
    for session in needed_sessions:
        print(f"loading NAV window for {session}...", flush=True)
        nav = load_nav(session_dirs[session], times_by_session[session])
        nav_by_session[session] = nav
        intervals = np.diff(nav.t)
        nav_summaries.append({"session": session, "nav_rows": len(nav.t), "nav_start_s": nav.t[0],
                              "nav_end_s": nav.t[-1], "nav_interval_median_ms": float(np.median(intervals) * 1000),
                              "nav_interval_p95_ms": float(np.percentile(intervals, 95) * 1000),
                              "nav_gap_max_ms": float(np.max(intervals) * 1000)})
        nav_course.extend({"session": session, "window_start_s": nav.t[0], "window_end_s": nav.t[-1],
                           **nav_course_metrics(nav, field)} for field in NAV_FIELDS)

    groups_by_field = {field: defaultdict(list) for field in NAV_FIELDS}
    nav_sync_rows = []
    parsed_boxes = matched_nav = rejected_nav = 0
    for idx, pair in enumerate(pairs, 1):
        timestamp = parse_ts(pair.lidar.stem)
        nav = nav_by_session[pair.session]
        pose = nav.interpolate(timestamp)
        if pose is None:
            rejected_nav += 1
            continue
        matched_nav += 1
        pose["enu"] = nav.to_enu(pose["lat"], pose["lon"], pose["alt"])
        label_path, boxes, _ = base.read_label_file(pair.label)
        parsed_boxes += len(boxes)
        nav_sync_rows.append({"session": pair.session, "timestamp": pair.lidar.stem,
                              "nearest_nav_delta_ms": pose["nearest_nav_delta_ms"],
                              "nav_bracket_gap_ms": pose["nav_bracket_gap_ms"],
                              "status": pose["Status"], "age": pose["Age"], "warning": pose["Warning"]})
        for box in boxes:
            if box.cls != "Car":
                continue
            common = {"segment": box.segment, "timestamp": box.timestamp, "timestamp_s": box.timestamp_s,
                      "trackid": box.trackid, "class": box.cls, "x": box.x, "y": box.y,
                      "z": box.z, "yaw": box.yaw, "numpoints": box.numpoints}
            for field in NAV_FIELDS:
                obs = add_world_pose(common, nav, pose, field)
                if obs is None:
                    continue
                key = (box.segment, box.trackid)
                groups_by_field[field][key].append(obs)
        if idx % 300 == 0 or idx == len(pairs):
            print(f"paired frames {idx}/{len(pairs)}; NAV matched {matched_nav}; parsed boxes {parsed_boxes}", flush=True)

    for groups in groups_by_field.values():
        for track in groups.values():
            estimate_motion(track)

    rows = candidate_table(groups_by_field)
    config = choose_config(rows)
    selected_groups = groups_by_field[config["nav_heading_field"]]
    resolved = resolve_track_headings(selected_groups, config)
    write_csv(OUT / "world_yaw_candidate_comparison.csv", rows)
    write_csv(OUT / "nav_pose_coverage.csv", nav_summaries)
    write_csv(OUT / "nav_heading_course_check.csv", nav_course)
    write_csv(OUT / "nav_lidar_sync.csv", nav_sync_rows)
    write_csv(OUT / "resolved_world_headings.csv", resolved)
    point_time = inspect_point_time(pairs)

    # Candidate plot: world-frame motion-axis agreement for both NAV heading fields.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [name for name, _ in YAW_CANDIDATES]
    x = np.arange(len(names))
    width = 0.36
    for offset, field in ((-width / 2, "Heading"), (width / 2, "Heading2")):
        values = [next((r.get("motion_mod180_median_deg") for r in rows
                        if r["nav_heading_field"] == field and r["yaw_candidate"] == name), np.nan)
                  for name in names]
        ax.bar(x + offset, values, width, label=field)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("World-frame motion vs box axis error (deg, modulo 180)")
    ax.set_title("Yaw candidates after NAV ego-motion compensation")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "world_yaw_candidate_motion_error.png", dpi=160)
    plt.close(fig)

    source_counts = Counter(row["heading_source"] for row in resolved)
    valid_rows = [row for row in resolved if row["heading_valid"]]
    baseline = {}
    baseline_path = OUT.parent / "track_yaw_candidate_summary.csv"
    if baseline_path.exists():
        with baseline_path.open("r", encoding="utf-8-sig", newline="") as stream:
            baseline = next((r for r in csv.DictReader(stream) if r.get("yaw_candidate") == "yaw"), {})
    heading_stats = next((r for r in nav_course if r["session"] == needed_sessions[0] and r["heading_field"] == "Heading"), {})
    output_report = f'''# World-frame yaw and heading audit

## Scope

- Raw and labels are read-only. All derived artifacts are written in `{OUT}`.
- Exact-timestamp matched label / `LIDAR_MERGED_EGO` frames: **{len(pairs):,}** across **{len({p.session for p in pairs})} sessions**.
- NAV interpolated successfully for **{matched_nav:,}** frames; rejected for NAV coverage/gap: **{rejected_nav:,}**.
- Parsed boxes in those frames: **{parsed_boxes:,}**; Car track observations were evaluated under both NAV `Heading` and `Heading2`.
- NAV timestamps were interpolated only when bracketing samples are no more than {MAX_NAV_GAP_S*1000:.0f} ms apart. `nav_lidar_sync.csv` contains per-frame deltas and NAV quality fields (`Status`, `Age`, `Warning`) for review; status codes are not interpreted here.
- Example point-time audit: `{point_time}`. Point timestamps are recorded but this audit does **not** deskew or rewrite any cloud.

## Method

1. Interpolate the timestamp-matched ego pose from `OTHERS/NAV`.
2. Convert GPS lat/lon/alt to session-local ENU and transform each ego-frame box center into ENU. NAV compass heading is converted to ENU yaw as `90 deg - heading`, consistent with ego `+X` forward and `+Y` left.
3. Compare `Heading` and `Heading2` on common support: both heading values and both standard deviations must be present, both standard deviations at most 2 deg, and each observation must pass the speed ({MIN_OBJECT_SPEED_MPS:.1f}-{MAX_OBJECT_SPEED_MPS:.0f} m/s) and ego-turn (<{MAX_EGO_TURN_DEG_S:.0f} deg/s) gates under both NAV fields. Then evaluate all six raw-yaw transforms modulo 180 degrees.
4. For the selected candidate, resolve the 180-degree branch only when motion is within {MAX_AXIS_MOTION_ERROR_DEG:.0f} deg of the box axis. Propagate that branch through contiguous observations of the same track, preserving each frame's measured box axis. Tracks without a confident moving anchor remain `heading_invalid`.

## Results

- NAV `Heading` vs compass course from `Ve/Vn` in the first session: median error **{heading_stats.get('course_error_median_deg')} deg**, P90 **{heading_stats.get('course_error_p90_deg')} deg** over {heading_stats.get('n_speed_ge_2mps')} rows with ego speed >=2 m/s in the loaded audit window. See `nav_heading_course_check.csv` for both heading fields and all loaded windows.
- Best-ranked world candidate: **NAV `{config['nav_heading_field']}` + box yaw `{config['yaw_candidate']}`**.
- Heading-field comparison uses the same **{config.get('common_moving_sample', 0):,}** moving observations for every NAV/yaw candidate. Best candidate results: median/P90 axis error **{config.get('motion_mod180_median_deg')} / {config.get('motion_mod180_p90_deg')} deg**; within 20/25 deg: **{config.get('motion_mod180_lt20_pct')}% / {config.get('motion_mod180_lt25_pct')}%**.
- Low-motion tracks with sufficient ego rotation for a static-axis check: **{config.get('static_tracks_with_ego_turn', 0)}**; median per-track world-axis jitter **{config.get('static_axis_jitter_track_median_deg')} deg**. This can help distinguish yaw sign without assuming a target car's velocity is always its forward direction.
- Resolved output rows: **{len(resolved):,}**; heading valid by direct world motion: **{source_counts.get('world_motion_assumed_forward', 0):,}**; propagated within track: **{source_counts.get('track_propagation', 0):,}**; still invalid: **{source_counts.get('heading_invalid', 0):,}**.
- Previous ego-frame track baseline for raw `yaw`: orientation-mod-180 median/P90 **{baseline.get('orientation_mod180_median_deg')} / {baseline.get('orientation_mod180_p90_deg')} deg** over {baseline.get('moving_car_pairs')} pairs. Compare with world-frame candidate metrics, noting filters and sample definitions differ.

## Interpretation and limits

- Ego-motion compensation is now applied to frame-to-frame box centers; the yaw audit is derived, not a source-data conversion.
- Motion-selected front/back assumes the target vehicle is driving forward. Reversing, lateral parking maneuvers, turns, or ID switches can produce a wrong sign; use the `heading_confidence` and `heading_source` columns and retain `heading_invalid` when evidence is weak.
- Propagation carries the sign branch, not a fixed absolute heading. Gaps over {MAX_PROPAGATION_GAP_S:.2f} s split a track segment.
- GPS position is used as the ego translation origin without applying a verified GNSS-antenna-to-LIDAR/ego lever arm. This can bias box-center velocity during ego turns; turn-rate filtering reduces but does not eliminate that risk. World yaw uses NAV orientation and is not affected by that translation offset.
- `Heading` and `Heading2` are both tested; empirical course agreement is not a substitute for the sensor vendor's field definition. NAV quality fields are preserved for manual filtering.
- Static objects without a moving anchor retain a valid box axis modulo 180 degrees where available, but have `heading_valid=false`; camera evidence is needed for front/back.
- Point-level motion compensation is separate. LAZ includes `point_timestamp`, but `LIDAR_MERGED_EGO` may already be deskewed. Do not apply another deskew without confirming the export convention and sensor-to-ego extrinsic.
- Candidate ranking is a diagnostic, not an automatic production label rewrite. Verify the winner on held-out tracks and camera overlays before adopting it.

## Artifacts

- `world_yaw_candidate_comparison.csv`
- `resolved_world_headings.csv`
- `nav_pose_coverage.csv`, `nav_heading_course_check.csv`, `nav_lidar_sync.csv`
- `world_yaw_candidate_motion_error.png`
'''
    (OUT / "world_yaw_summary.md").write_text(output_report, encoding="utf-8")
    print(f"World yaw audit complete: {OUT}", flush=True)


if __name__ == "__main__":
    main()
