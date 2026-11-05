"""Deskew one TOP scan and compare its Rider yaw convention with track motion."""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import date
from pathlib import Path

import cv2
import laspy
import numpy as np

from pnk_comet_autolabel.calibration import load_calibration
from pnk_comet_autolabel.constants import CAMERA_MAP
from pnk_comet_autolabel.reader import nearest_path, timestamp_ns
from pnk_comet_autolabel.sensor_preprocessing import (
    NavTimeline, _camera_files, _point_timestamps_ns, _read_nav,
)
from .heading import build_session_heading_audit


def _quat_rotation(q):
    w, x, y, z = map(float, q)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _wrap_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _angle_err_mod_pi(a, b):
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


def _read_boxes(path):
    out = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        f = line.split()
        if len(f) < 11:
            continue
        try:
            vals = list(map(float, f[:7]))
        except ValueError:
            continue
        out[(f[7], f[10])] = {"x": vals[0], "y": vals[1], "z": vals[2],
                              "length": vals[3], "width": vals[4],
                              "height": vals[5], "yaw": vals[6],
                              "class": f[7], "track_id": f[10]}
    return out


def _corners(center, length, width, height, yaw):
    local = np.asarray([[sx * length / 2, sy * width / 2, sz * height / 2]
                        for sz in (-1, 1) for sx in (-1, 1) for sy in (-1, 1)],
                       dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return local @ rz.T + np.asarray(center, dtype=np.float64)


def _draw_bev(points, center, target, axes, lidar_path, label_paths, output):
    w, h, scale = 1150, 1120, 25.0
    cx, bottom = w // 2, h - 72
    canvas = np.full((h, w, 3), (16, 19, 23), np.uint8)
    for x in range(-10, 36, 5):
        v = int(bottom - x * scale)
        cv2.line(canvas, (50, v), (w - 30, v), (45, 49, 54), 1)
        cv2.putText(canvas, f"{x}m", (w - 65, v - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (135, 140, 145), 1, cv2.LINE_AA)
    for y in range(-20, 21, 5):
        u = int(cx - y * scale)
        cv2.line(canvas, (u, 32), (u, h - 40), (42, 46, 51), 1)
    selected = points[(points[:, 0] > -10) & (points[:, 0] < 36) &
                      (points[:, 1] > -20) & (points[:, 1] < 20) &
                      (points[:, 2] > -3) & (points[:, 2] < 3)]
    if len(selected) > 70000:
        selected = selected[::int(math.ceil(len(selected) / 70000))]
    for x, y, z in selected:
        u, v = int(cx - y * scale), int(bottom - x * scale)
        if 0 <= u < w and 0 <= v < h:
            shade = int(np.clip(95 + 28 * z, 65, 200))
            cv2.circle(canvas, (u, v), 1, (shade, shade, shade), -1)

    # The source/global convention and segment-local candidate get separate
    # boxes so the point cloud shows which interpretation follows the object.
    def draw_box(yaw, color, thickness):
        center_3d = (center[0], center[1], target["z"])
        c = _corners(center_3d, target["length"], target["width"], target["height"], yaw)
        px = [(int(cx - p[1] * scale), int(bottom - p[0] * scale)) for p in c]
        for a, b in ((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6),
                     (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)):
            cv2.line(canvas, px[a], px[b], color, thickness, cv2.LINE_AA)
    draw_box(target["raw_yaw"], (110, 110, 125), 1)
    draw_box(axes["candidate_local_axis"], (0, 220, 255), 3)
    cu, cv = int(cx - center[1] * scale), int(bottom - center[0] * scale)
    cv2.drawMarker(canvas, (cu, cv), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

    def arrow(theta, color, label, length=3.1, offset_y=0.0):
        x, y = center[0], center[1] + offset_y
        start = (int(cx - y * scale), int(bottom - x * scale))
        end = (int(cx - (y + length * math.sin(theta)) * scale),
               int(bottom - (x + length * math.cos(theta)) * scale))
        cv2.arrowedLine(canvas, start, end, color, 4, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(canvas, label, (end[0] + 5, end[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 2, cv2.LINE_AA)
    arrow(axes["global_yaw_local"], (0, 115, 255), "Heading2 + yaw", offset_y=-0.14)
    arrow(axes["resolved_local"], (80, 245, 80), "local candidate + front/back", offset_y=0.14)
    arrow(axes["motion_local"], (255, 220, 0), "track motion", length=2.1)

    cv2.rectangle(canvas, (0, 0), (w, 140), (10, 12, 16), -1)
    cv2.putText(canvas, f"LIDAR_TOP | {target['class']} track {target['track_id']} | {lidar_path.name}",
                (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.69, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"box center interpolated from labels {label_paths[0].name} and {label_paths[1].name}",
                (22, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (215, 220, 225), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"source candidate axis error={axes['global_error_deg']:.1f} deg; local candidate error={axes['candidate_error_deg']:.1f} deg; motion={axes['speed_mps']:.2f} m/s",
                (22, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (215, 220, 225), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Gray box=source yaw; yellow box=segment-local yaw candidate | Orange=Heading2+yaw | Green=motion-resolved candidate | Cyan=motion",
                (40, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def _project_camera_box(points_world, center_world, arrow_world_angles,
                        image_path, channel, cam_ns, nav, ego_to_camera,
                        intrinsic, distortion):
    pose = nav.at(np.asarray([cam_ns], dtype=np.int64))
    if not bool(pose.valid[0]):
        return None
    R = pose.rotation_world_ego[0]
    origin = pose.position_enu[0]

    def to_camera(world):
        p_ego = (np.asarray(world) - origin) @ R
        return p_ego @ ego_to_camera[:3, :3].T + ego_to_camera[:3, 3]

    cam_points = to_camera(points_world)
    uv, _ = cv2.projectPoints(cam_points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                              intrinsic, distortion)
    uv = uv.reshape(-1, 2)
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    h, w = image.shape[:2]
    vis = np.zeros(len(uv), dtype=bool)
    vis[:] = cam_points[:, 2] > 0.1
    edges = ((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6),
             (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7))
    for a, b in edges:
        if not (vis[a] and vis[b]):
            continue
        pa, pb = tuple(np.rint(uv[a]).astype(int)), tuple(np.rint(uv[b]).astype(int))
        if (-w <= pa[0] < 2 * w and -h <= pa[1] < 2 * h and
                -w <= pb[0] < 2 * w and -h <= pb[1] < 2 * h):
            cv2.line(image, pa, pb, (0, 220, 255), 3, cv2.LINE_AA)
    center_cam = to_camera(np.asarray(center_world)[None])[0]
    arrow_points = [center_cam]
    for angle in arrow_world_angles:
        end_world = center_world + np.asarray([3 * math.cos(angle), 3 * math.sin(angle), 0])
        arrow_points.append(to_camera(end_world[None])[0])
    arrow_uv, _ = cv2.projectPoints(np.asarray(arrow_points).reshape(-1, 1, 3),
                                    np.zeros(3), np.zeros(3), intrinsic, distortion)
    arrow_uv = arrow_uv.reshape(-1, 2).astype(int)
    if center_cam[2] > 0.1:
        for idx, color in ((1, (0, 115, 255)), (2, (80, 245, 80)), (3, (255, 220, 0))):
            if arrow_points[idx][2] > 0.1:
                cv2.arrowedLine(image, tuple(arrow_uv[0]), tuple(arrow_uv[idx]),
                                color, 3, cv2.LINE_AA, tipLength=0.15)
    thumbnail = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
    return thumbnail, image


def visualize_one(raw_root: Path, label_dir: Path, calibration_dir: Path,
                  lidar_path: Path, track_id: str, class_name: str,
                  output_dir: Path) -> dict:
    raw_root, label_dir, lidar_path = raw_root.resolve(), label_dir.resolve(), lidar_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_ns = timestamp_ns(lidar_path)
    target_s = target_ns / 1e9
    label_items = sorted(((timestamp_ns(p), p) for p in label_dir.glob("*.txt")), key=lambda q: q[0])
    before = [x for x in label_items if x[0] < target_ns]
    after = [x for x in label_items if x[0] > target_ns]
    if not before or not after:
        raise RuntimeError("The TOP scan must be bracketed by 3D label frames")
    (t0, p0), (t1, p1) = before[-1], after[0]
    b0, b1 = _read_boxes(p0), _read_boxes(p1)
    key = (class_name, str(track_id))
    if key not in b0 or key not in b1:
        raise RuntimeError(f"Track {class_name}/{track_id} is not present in both neighboring labels")
    box0, box1 = b0[key], b1[key]
    alpha = (target_ns - t0) / (t1 - t0)

    cloud = laspy.read(lidar_path)
    if "point_timestamp" not in set(cloud.point_format.dimension_names):
        raise RuntimeError("LIDAR_TOP cloud has no point_timestamp dimension")
    point_ns, point_time_mode = _point_timestamps_ns(
        np.asarray(cloud["point_timestamp"]), target_ns, "auto")
    cameras = _camera_files(raw_root)
    camera_matches = {src: nearest_path(files, target_ns) for src, files in cameras.items() if files}
    all_times = [int(point_ns.min()), int(point_ns.max()), t0, t1, target_ns]
    all_times += [match[0] for match in camera_matches.values()]
    nav_rows = _read_nav(raw_root, min(all_times), max(all_times))
    nav = NavTimeline(nav_rows, max_gap_ms=50.0)

    # Deskew the requested TOP scan exactly like the full preprocessing path.
    _, extrinsics, calibration_period = load_calibration(
        calibration_dir, "VF6_01", date(2026, 3, 2))
    T_ego_top = np.asarray(extrinsics["LIDAR_TOP"], dtype=np.float64)
    p_top = np.column_stack((np.asarray(cloud.x, dtype=np.float64),
                             np.asarray(cloud.y, dtype=np.float64),
                             np.asarray(cloud.z, dtype=np.float64)))
    p_ego_i = p_top @ T_ego_top[:3, :3].T + T_ego_top[:3, 3]
    point_pose, end_pose = nav.at(point_ns), nav.at(target_ns)
    if not np.all(point_pose.valid) or not bool(end_pose.valid[0]):
        raise RuntimeError("NAV coverage is insufficient to deskew the requested scan")
    p_world = np.einsum("nij,nj->ni", point_pose.rotation_world_ego, p_ego_i) + point_pose.position_enu
    points_ego_end = (p_world - end_pose.position_enu[0]) @ end_pose.rotation_world_ego[0]

    # Recompute candidate diagnostics for this complete annotation segment.
    heading_summary = build_session_heading_audit(
        raw_root, label_dir, output_dir / "segment_heading_audit")
    heading_csv = output_dir / "segment_heading_audit" / "resolved_world_headings.csv"
    with heading_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        audit_rows = list(csv.DictReader(stream))
    row0 = next(r for r in audit_rows if r["timestamp"] == p0.stem and
                r["trackid"] == str(track_id) and r["class"] == class_name)
    row1 = next(r for r in audit_rows if r["timestamp"] == p1.stem and
                r["trackid"] == str(track_id) and r["class"] == class_name)
    if not row0["motion_heading_world_deg"] or not row1["motion_heading_world_deg"]:
        raise RuntimeError("Track has no usable motion direction at the bracketing labels")

    pose0, pose1 = nav.at(t0), nav.at(t1)
    pose_target = end_pose
    def center_world(box, pose):
        local = np.asarray([box["x"], box["y"], box["z"]], dtype=np.float64)
        return local @ pose.rotation_world_ego[0].T + pose.position_enu[0]
    world0, world1 = center_world(box0, pose0), center_world(box1, pose1)
    world_center = (1 - alpha) * world0 + alpha * world1
    center_target = (world_center - pose_target.position_enu[0]) @ pose_target.rotation_world_ego[0]
    center_target[2] = (1 - alpha) * box0["z"] + alpha * box1["z"]
    yaw_raw = (1 - alpha) * box0["yaw"] + alpha * box1["yaw"]
    length = (1 - alpha) * box0["length"] + alpha * box1["length"]
    width = (1 - alpha) * box0["width"] + alpha * box1["width"]
    height = (1 - alpha) * box0["height"] + alpha * box1["height"]
    motion0 = math.radians(float(row0["motion_heading_world_deg"]))
    motion1 = math.radians(float(row1["motion_heading_world_deg"]))
    motion_world = math.atan2((1 - alpha) * math.sin(motion0) + alpha * math.sin(motion1),
                              (1 - alpha) * math.cos(motion0) + alpha * math.cos(motion1))
    ego_yaw = math.atan2(pose_target.rotation_world_ego[0, 1, 0],
                         pose_target.rotation_world_ego[0, 0, 0])
    candidate_info = heading_summary["heading_classes"][class_name]["best_local_candidate"]
    candidate_name = candidate_info["yaw_candidate"]
    yaw_fn = dict(__import__("world_yaw_audit").YAW_CANDIDATES)[candidate_name]
    candidate_local_axis = yaw_fn(yaw_raw)
    source_world = ego_yaw + yaw_raw
    candidate_world_axis = ego_yaw + candidate_local_axis
    motion_err_source = _angle_err_mod_pi(source_world, motion_world)
    motion_err_candidate = _angle_err_mod_pi(candidate_world_axis, motion_world)
    resolved_world = candidate_world_axis
    if abs(_wrap_pi(resolved_world + math.pi - motion_world)) < abs(_wrap_pi(resolved_world - motion_world)):
        resolved_world += math.pi
    axes = {
        "global_yaw_local": yaw_raw,
        "candidate_local_axis": candidate_local_axis,
        "resolved_local": resolved_world - ego_yaw,
        "motion_local": motion_world - ego_yaw,
        "global_error_deg": math.degrees(motion_err_source),
        "candidate_error_deg": math.degrees(motion_err_candidate),
        "speed_mps": (1 - alpha) * float(row0["world_speed_mps"]) + alpha * float(row1["world_speed_mps"]),
    }
    target = {**box0, "x": float(center_target[0]), "y": float(center_target[1]),
              "z": float(center_target[2]), "length": length, "width": width,
              "height": height, "raw_yaw": yaw_raw}
    lidar_png = output_dir / f"{lidar_path.stem}_{class_name.lower()}_{track_id}_lidar.png"
    _draw_bev(points_ego_end, (target["x"], target["y"]), target, axes,
              lidar_path, [p0, p1], lidar_png)

    local_candidate_corners = _corners(
        world_center, length, width, height,
        candidate_world_axis)
    camera_tiles, camera_names, camera_overlay_files = [], [], []
    intrinsics, extrinsics, _ = load_calibration(calibration_dir, "VF6_01", date(2026, 3, 2))
    for source, channel in CAMERA_MAP.items():
        if source not in camera_matches or source not in intrinsics or source not in extrinsics:
            continue
        cam_ns, cam_path = camera_matches[source]
        projected = _project_camera_box(
            local_candidate_corners, world_center,
            (source_world, resolved_world, motion_world), cam_path, channel, cam_ns,
            nav, np.asarray(extrinsics[source], dtype=np.float64),
            np.asarray(intrinsics[source]["camera_matrix"], dtype=np.float64),
            np.asarray(intrinsics[source]["distortion_coefficients"], dtype=np.float64).reshape(-1, 1))
        if projected is None:
            continue
        image, full_image = projected
        label = f"{channel} | {cam_path.name} | {(cam_ns-target_ns)/1e6:+.2f}ms"
        camera_tiles.append((channel, image, label))
        overlay_path = output_dir / f"{lidar_path.stem}_{class_name.lower()}_{track_id}_{channel.lower()}_overlay.jpg"
        cv2.imwrite(str(overlay_path), full_image)
        camera_overlay_files.append(str(overlay_path))
        camera_names.append({"channel": channel, "file": str(cam_path),
                             "delta_ms": (cam_ns - target_ns) / 1e6})
    tile_h, tile_w = 300, 480
    sheet = np.zeros((70 + 2 * tile_h, 4 * tile_w, 3), np.uint8)
    cv2.putText(sheet, "Exact LiDAR scan's nearest raw camera frames | yellow box=local candidate; orange=Heading2+yaw; green=resolved; cyan=motion",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(sheet, "Projected box uses the label track interpolated between the two 5 Hz annotations",
                (12, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (195, 205, 215), 1, cv2.LINE_AA)
    for i, (_, image, label) in enumerate(camera_tiles):
        row, col = divmod(i, 4)
        y, x = 70 + row * tile_h, col * tile_w
        tile = np.zeros((tile_h, tile_w, 3), np.uint8)
        tile[:270] = image
        cv2.putText(tile, label, (4, 289), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (235, 235, 235), 1, cv2.LINE_AA)
        sheet[y:y + tile_h, x:x + tile_w] = tile
    camera_png = output_dir / f"{lidar_path.stem}_{class_name.lower()}_{track_id}_cameras.png"
    cv2.imwrite(str(camera_png), sheet)

    result = {
        "lidar_file": str(lidar_path), "lidar_timestamp_ns": target_ns,
        "label_before": str(p0), "label_after": str(p1),
        "box_interpolation_fraction": alpha, "track_id": str(track_id), "class": class_name,
        "candidate_from_segment_diagnostic": candidate_name,
        "segment_rider_candidate_summary": candidate_info,
        "source_interpretation_heading2_plus_yaw_axis_error_deg": axes["global_error_deg"],
        "local_candidate_axis_error_deg": axes["candidate_error_deg"],
        "local_candidate_resolved_world_heading_deg": math.degrees(resolved_world) % 360,
        "interpolated_track_motion_world_heading_deg": math.degrees(motion_world) % 360,
        "interpolated_speed_mps": axes["speed_mps"],
        "point_timestamp_mode": point_time_mode,
        "lidar_deskew_points": len(points_ego_end),
        "calibration_period": calibration_period,
        "camera_files": camera_names,
        "camera_overlay_files": camera_overlay_files,
        "lidar_visualization": str(lidar_png),
        "camera_contact_sheet": str(camera_png),
        "raw_data_modified": False,
        "note": "No exact-time box annotation exists for this 10 Hz scan; box center/dimensions/yaw are interpolated from the bracketing 5 Hz labels in ENU/world coordinates.",
    }
    (output_dir / "heading_example_metadata.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--lidar", type=Path, required=True)
    ap.add_argument("--track-id", required=True)
    ap.add_argument("--class-name", choices=("Car", "Rider"), default="Rider")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    visualize_one(args.raw, args.labels, args.calibration, args.lidar,
                  args.track_id, args.class_name, args.out)


if __name__ == "__main__":
    main()
