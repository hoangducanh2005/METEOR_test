"""Render all annotated objects and motion-qualified headings over one TOP scan."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2
import laspy
import numpy as np

from pnk_comet_autolabel.calibration import load_calibration
from pnk_comet_autolabel.reader import timestamp_ns
from pnk_comet_autolabel.sensor_preprocessing import (
    NavTimeline, _point_timestamps_ns, _read_nav,
)


def _parse_ts(text: str) -> float:
    sec, frac = text.split("-", 1)
    return int(sec) + int(frac) / 1_000_000_000.0


def _wrap_pi(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def _axis_error(a: float, b: float) -> float:
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


def _parse_boxes(label_path: Path) -> list[dict]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        f = line.split()
        if len(f) < 11:
            continue
        try:
            vals = list(map(float, f[:7]))
        except ValueError:
            continue
        boxes.append({"x": vals[0], "y": vals[1], "z": vals[2],
                      "length": vals[3], "width": vals[4], "height": vals[5],
                      "yaw": vals[6], "class": f[7], "track": f[10]})
    return boxes


def _local_to_world(box: dict, pose) -> tuple[np.ndarray, float]:
    yaw = math.atan2(pose.rotation_world_ego[0, 1, 0],
                     pose.rotation_world_ego[0, 0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    xy = pose.position_enu[0, :2] + np.asarray([
        c * box["x"] - s * box["y"],
        s * box["x"] + c * box["y"],
    ])
    return xy, yaw


def _track_state(samples: list[dict]) -> tuple[str, float, float]:
    samples = sorted(samples, key=lambda row: row["t"])
    pairs = []
    for a, b in zip(samples, samples[1:]):
        dt = b["t"] - a["t"]
        if 0.05 <= dt <= 1.0:
            pairs.append(float(np.linalg.norm(b["world_xy"] - a["world_xy"]) / dt))
    if len(samples) < 4 or samples[-1]["t"] - samples[0]["t"] < 0.6 or len(pairs) < 3:
        return "UNKNOWN", float("nan"), float("nan")
    median_speed = float(np.median(pairs))
    p90_speed = float(np.percentile(pairs, 90))
    center = np.median(np.stack([row["world_xy"] for row in samples]), axis=0)
    spread_p95 = float(np.percentile(
        [np.linalg.norm(row["world_xy"] - center) for row in samples], 95))
    if median_speed <= 0.5 and p90_speed <= 1.0 and spread_p95 <= 0.75:
        return "STATIC", median_speed, spread_p95
    if median_speed >= 1.5:
        return "MOVING", median_speed, spread_p95
    return "UNKNOWN", median_speed, spread_p95


def _local_velocity(samples: list[dict], target_s: float) -> tuple[np.ndarray | None, float]:
    nearby = [row for row in samples if abs(row["t"] - target_s) <= 0.8]
    velocities = []
    for i, a in enumerate(nearby):
        for b in nearby[i + 1:]:
            dt = b["t"] - a["t"]
            if 0.15 <= dt <= 0.8:
                velocities.append((b["world_xy"] - a["world_xy"]) / dt)
    if not velocities:
        return None, float("nan")
    velocity = np.median(np.stack(velocities), axis=0)
    return velocity, float(np.linalg.norm(velocity))


def _box_corners_xy(box: dict) -> np.ndarray:
    local = np.asarray([[-box["length"] / 2, -box["width"] / 2],
                        [-box["length"] / 2, box["width"] / 2],
                        [box["length"] / 2, box["width"] / 2],
                        [box["length"] / 2, -box["width"] / 2]])
    c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
    rot = np.asarray([[c, -s], [s, c]])
    return local @ rot.T + np.asarray([box["x"], box["y"]])


def _render(points: np.ndarray, objects: list[dict], output: Path,
            lidar_name: str, target_ns: int, static_count: int,
            moving_count: int, unknown_count: int) -> None:
    width_px, height_px, scale = 1500, 1620, 22.0
    cx, top, xmax = width_px // 2, 155, 50.0
    canvas = np.full((height_px, width_px, 3), (17, 20, 24), np.uint8)

    def pixel(x: float, y: float) -> tuple[int, int]:
        return int(cx - y * scale), int(top + (xmax - x) * scale)

    for x in range(-10, 51, 5):
        v = pixel(x, 0)[1]
        if 0 <= v < height_px:
            cv2.line(canvas, (45, v), (width_px - 35, v), (46, 50, 56), 1)
            cv2.putText(canvas, f"{x}m", (width_px - 75, v - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (135, 140, 148), 1, cv2.LINE_AA)
    for y in range(-30, 31, 5):
        u = pixel(0, y)[0]
        cv2.line(canvas, (u, 125), (u, height_px - 35), (42, 46, 52), 1)

    keep = ((points[:, 0] >= -10) & (points[:, 0] <= 50) &
            (points[:, 1] >= -30) & (points[:, 1] <= 30) &
            (points[:, 2] >= -3.0) & (points[:, 2] <= 4.0))
    visible_points = points[keep]
    if len(visible_points) > 100000:
        visible_points = visible_points[::int(math.ceil(len(visible_points) / 100000))]
    for x, y, z in visible_points:
        u, v = pixel(float(x), float(y))
        if 0 <= u < width_px and 0 <= v < height_px:
            shade = int(np.clip(105 + z * 25, 68, 205))
            cv2.circle(canvas, (u, v), 1, (shade, shade, shade), -1)

    colors = {"Car": (0, 165, 255), "Rider": (255, 175, 60),
              "Pedestrian": (220, 90, 235)}
    visible_boxes = 0
    visible_states = {"STATIC": 0, "MOVING": 0, "UNKNOWN": 0}
    for obj in objects:
        if not (-12 <= obj["x"] <= 52 and abs(obj["y"]) <= 32):
            continue
        visible_boxes += 1
        visible_states[obj["state"]] += 1
        color = colors.get(obj["class"], (165, 175, 185))
        if obj["state"] == "STATIC":
            color = tuple(int(channel * 0.65) for channel in color)
        poly = np.asarray([pixel(float(x), float(y))
                           for x, y in _box_corners_xy(obj)], np.int32)
        cv2.polylines(canvas, [poly], True, color, 2 if obj["heading_local"] is not None else 1,
                      cv2.LINE_AA)
        center = pixel(obj["x"], obj["y"])
        cv2.circle(canvas, center, 3, color, -1, cv2.LINE_AA)
        if obj["heading_local"] is not None:
            theta = obj["heading_local"]
            end = pixel(obj["x"] + 2.1 * math.cos(theta),
                        obj["y"] + 2.1 * math.sin(theta))
            cv2.arrowedLine(canvas, center, end, (80, 245, 80), 3,
                            cv2.LINE_AA, tipLength=0.22)
            state_tag = "H"
        elif obj["state"] == "STATIC":
            state_tag = "S"
        else:
            state_tag = "?"
        label = f"{obj['class'][0]}{obj['track']}:{state_tag}"
        cv2.putText(canvas, label, (center[0] + 4, center[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (245, 245, 245), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (width_px, 122), (10, 12, 16), -1)
    stamp = f"{target_ns // 1_000_000_000}-{target_ns % 1_000_000_000:09d}"
    cv2.putText(canvas, f"LIDAR_TOP top-down BEV | {lidar_name}", (22, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"scan={stamp} | visible labeled boxes={visible_boxes} | static={visible_states['STATIC']}, moving={visible_states['MOVING']}, uncertain={visible_states['UNKNOWN']}",
                (22, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (215, 220, 225), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Green arrow=heading resolved from world-frame motion; S=static, heading omitted; ?=unresolved. Car/Rider/Pedestrian box colors differ.",
                (22, 99), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (215, 220, 225), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def render_scan(raw_root: Path, label_dir: Path, calibration_dir: Path,
                lidar_path: Path, output: Path) -> dict:
    raw_root, label_dir, lidar_path = raw_root.resolve(), label_dir.resolve(), lidar_path.resolve()
    target_ns = timestamp_ns(lidar_path)
    target_s = target_ns / 1e9
    label_paths = sorted(label_dir.glob("*.txt"), key=lambda p: timestamp_ns(p))
    if not label_paths:
        raise FileNotFoundError(f"No labels found in {label_dir}")
    labels = [(timestamp_ns(path), path, _parse_boxes(path)) for path in label_paths]
    before = [item for item in labels if item[0] <= target_ns]
    after = [item for item in labels if item[0] >= target_ns]
    if not before or not after:
        raise RuntimeError("Target LiDAR timestamp is not bracketed by labels")
    left, right = before[-1], after[0]

    cloud = laspy.read(lidar_path)
    if "point_timestamp" not in set(cloud.point_format.dimension_names):
        raise RuntimeError("LIDAR_TOP cloud has no point_timestamp dimension")
    point_ns, point_mode = _point_timestamps_ns(
        np.asarray(cloud["point_timestamp"]), target_ns, "auto")
    all_times = [point_ns.min(), point_ns.max(), target_ns]
    all_times += [item[0] for item in labels]
    nav_rows = _read_nav(raw_root, int(min(all_times)), int(max(all_times)))
    nav = NavTimeline(nav_rows, max_gap_ms=50.0)
    _, extrinsics, calibration_period = load_calibration(
        calibration_dir, "VF6_01", date(2026, 3, 2))
    T_ego_top = np.asarray(extrinsics["LIDAR_TOP"], dtype=np.float64)
    p_top = np.column_stack((np.asarray(cloud.x, dtype=np.float64),
                             np.asarray(cloud.y, dtype=np.float64),
                             np.asarray(cloud.z, dtype=np.float64)))
    p_ego_i = p_top @ T_ego_top[:3, :3].T + T_ego_top[:3, 3]
    point_pose, target_pose = nav.at(point_ns), nav.at(target_ns)
    if not np.all(point_pose.valid) or not bool(target_pose.valid[0]):
        raise RuntimeError("Insufficient NAV coverage to deskew the LiDAR scan")
    p_world = (np.einsum("nij,nj->ni", point_pose.rotation_world_ego, p_ego_i)
               + point_pose.position_enu)
    points_target = ((p_world - target_pose.position_enu[0])
                     @ target_pose.rotation_world_ego[0])
    target_ego_yaw = math.atan2(target_pose.rotation_world_ego[0, 1, 0],
                                target_pose.rotation_world_ego[0, 0, 0])

    tracks: dict[tuple[str, str], list[dict]] = defaultdict(list)
    keyed_frames = []
    for stamp_ns, path, boxes in labels:
        pose = nav.at(stamp_ns)
        if not bool(pose.valid[0]):
            continue
        keyed = {}
        for box in boxes:
            world_xy, ego_yaw = _local_to_world(box, pose)
            row = {**box, "t": stamp_ns / 1e9, "stamp_ns": stamp_ns,
                   "world_xy": world_xy, "ego_yaw": ego_yaw}
            key = (box["class"], str(box["track"]))
            tracks[key].append(row)
            keyed[key] = row
        keyed_frames.append((stamp_ns, path, keyed))

    states = {key: _track_state(samples) for key, samples in tracks.items()}
    left_keyed = next((entry[2] for entry in keyed_frames if entry[0] == left[0]), {})
    right_keyed = next((entry[2] for entry in keyed_frames if entry[0] == right[0]), {})
    alpha = 0.0 if right[0] == left[0] else (target_ns - left[0]) / (right[0] - left[0])
    objects = []
    for key in sorted(set(left_keyed) | set(right_keyed)):
        a, b = left_keyed.get(key), right_keyed.get(key)
        if a is not None and b is not None:
            world_xy = (1 - alpha) * a["world_xy"] + alpha * b["world_xy"]
            box = {field: (1 - alpha) * a[field] + alpha * b[field]
                   for field in ("length", "width", "height", "z")}
            yaw_delta = _wrap_pi(b["yaw"] - a["yaw"])
            box["yaw"] = _wrap_pi(a["yaw"] + alpha * yaw_delta)
            box["class"], box["track"] = key
        else:
            row = a if a is not None else b
            world_xy = row["world_xy"]
            box = {k: row[k] for k in
                   ("length", "width", "height", "z", "yaw", "class", "track")}
        dx, dy = world_xy - target_pose.position_enu[0, :2]
        c, s = math.cos(target_ego_yaw), math.sin(target_ego_yaw)
        x = c * dx + s * dy
        y = -s * dx + c * dy
        state, whole_track_median_speed, spread = states.get(key, ("UNKNOWN", float("nan"), float("nan")))
        samples = tracks.get(key, [])
        velocity, local_speed = _local_velocity(samples, target_s)
        heading_world = None
        if state != "STATIC" and velocity is not None and local_speed >= 1.5:
            motion_world = math.atan2(float(velocity[1]), float(velocity[0]))
            axis_world = _wrap_pi(target_ego_yaw + box["yaw"])
            if _axis_error(axis_world, motion_world) <= math.radians(30.0):
                heading_world = axis_world
                if abs(_wrap_pi(axis_world + math.pi - motion_world)) < abs(_wrap_pi(axis_world - motion_world)):
                    heading_world = _wrap_pi(axis_world + math.pi)
        heading_local = (_wrap_pi(heading_world - target_ego_yaw)
                         if heading_world is not None else None)
        objects.append({**box, "x": float(x), "y": float(y), "state": state,
                        "heading_local": heading_local,
                        "track_median_speed_mps": whole_track_median_speed,
                        "static_spread_p95_m": spread,
                        "local_speed_mps": local_speed})

    static_n = sum(row["state"] == "STATIC" for row in objects)
    moving_n = sum(row["state"] == "MOVING" for row in objects)
    unknown_n = sum(row["state"] == "UNKNOWN" for row in objects)
    output.parent.mkdir(parents=True, exist_ok=True)
    _render(points_target, objects, output, lidar_path.name, target_ns,
            static_n, moving_n, unknown_n)
    return {"output": str(output), "label_before": str(left[1]),
            "label_after": str(right[1]), "interpolation_fraction": alpha,
            "visible_annotation_count": len(objects), "static_tracks": static_n,
            "moving_tracks": moving_n, "uncertain_tracks": unknown_n,
            "heading_count": sum(row["heading_local"] is not None for row in objects),
            "deskew_point_count": len(points_target), "point_timestamp_mode": point_mode,
            "calibration_period": calibration_period}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--lidar", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(render_scan(args.raw, args.labels, args.calibration, args.lidar, args.out))


if __name__ == "__main__":
    main()
