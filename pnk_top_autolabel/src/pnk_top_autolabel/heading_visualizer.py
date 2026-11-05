"""Visualize one heading correction in LiDAR BEV and synchronized cameras."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import laspy
import numpy as np


def _quat_rotation(q):
    w, x, y, z = map(float, q)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _box_corners(x, y, z, length, width, height, yaw):
    local = np.asarray([[sx * length / 2, sy * width / 2, sz * height / 2]
                        for sz in (-1, 1) for sx in (-1, 1) for sy in (-1, 1)],
                       dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return local @ rotation.T + np.asarray([x, y, z], dtype=np.float64)


def _pixel(point, K):
    if not np.isfinite(point).all() or point[2] <= 0.1:
        return None
    p = K @ point
    return int(round(p[0] / p[2])), int(round(p[1] / p[2]))


def _arrow_bev(canvas, center, angle_local, color, scale, cx, bottom, label,
               length_m=2.3, offset_y=0.0):
    x, y = center
    start = (int(cx - (y + offset_y) * scale), int(bottom - x * scale))
    end = (int(cx - (y + offset_y + length_m * math.sin(angle_local)) * scale),
           int(bottom - (x + length_m * math.cos(angle_local)) * scale))
    cv2.arrowedLine(canvas, start, end, color, 4, cv2.LINE_AA, tipLength=0.22)
    cv2.putText(canvas, label, (end[0] + 8, end[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _render_lidar(lidar_path, boxes, target, audit, out_path):
    cloud = laspy.read(lidar_path)
    xyz = np.column_stack((np.asarray(cloud.x), np.asarray(cloud.y), np.asarray(cloud.z)))
    width_px, height_px = 1050, 1100
    scale = 22.0
    cx, bottom = width_px // 2, height_px - 70
    canvas = np.zeros((height_px, width_px, 3), np.uint8)
    canvas[:] = (17, 20, 24)
    # Ego-frame BEV window: forward +X is up; left +Y is toward screen-left.
    for x in range(-10, 31, 5):
        v = int(bottom - x * scale)
        if 0 <= v < height_px:
            cv2.line(canvas, (70, v), (width_px - 35, v), (48, 52, 58), 1)
            cv2.putText(canvas, f"{x}m", (width_px - 68, v - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 145, 150), 1, cv2.LINE_AA)
    for y in range(-20, 21, 5):
        u = int(cx - y * scale)
        cv2.line(canvas, (u, 35), (u, height_px - 45), (42, 47, 52), 1)
    keep = ((xyz[:, 0] >= -10) & (xyz[:, 0] <= 30) &
            (xyz[:, 1] >= -20) & (xyz[:, 1] <= 20) &
            (xyz[:, 2] >= -3.0) & (xyz[:, 2] <= 3.0))
    pts = xyz[keep]
    if len(pts) > 50000:
        pts = pts[::int(math.ceil(len(pts) / 50000))]
    for x, y, z in pts:
        u, v = int(cx - y * scale), int(bottom - x * scale)
        if 0 <= u < width_px and 0 <= v < height_px:
            intensity = int(np.clip(95 + 35 * z, 65, 210))
            cv2.circle(canvas, (u, v), 1, (intensity, intensity, intensity), -1)

    target_id = (target["track_id"], target["class"])
    for box in boxes:
        x, y, z, length, width, height, yaw, cls, track = box
        corners = _box_corners(x, y, z, length, width, height, yaw)
        points = [(int(cx - p[1] * scale), int(bottom - p[0] * scale)) for p in corners]
        is_target = (str(track), cls) == target_id
        color = (0, 220, 255) if is_target else (90, 110, 120)
        thickness = 3 if is_target else 1
        for a, b in ((0, 1), (0, 2), (1, 3), (2, 3),
                     (4, 5), (4, 6), (5, 7), (6, 7),
                     (0, 4), (1, 5), (2, 6), (3, 7)):
            cv2.line(canvas, points[a], points[b], color, thickness, cv2.LINE_AA)
    center = (float(target["x"]), float(target["y"]))
    ego_yaw = math.radians(90.0 - float(audit["nav_heading_deg"]))
    raw_local = float(target["yaw"])
    heading_local = math.radians(float(audit["heading_world_deg"])) - ego_yaw
    motion_local = math.radians(float(audit["motion_heading_world_deg"])) - ego_yaw
    _arrow_bev(canvas, center, raw_local, (0, 125, 255), scale, cx, bottom,
               "raw directed yaw", offset_y=-0.10)
    _arrow_bev(canvas, center, heading_local, (80, 245, 80), scale, cx, bottom,
               "resolved heading", offset_y=0.10)
    _arrow_bev(canvas, center, motion_local, (255, 220, 0), scale, cx, bottom,
               "track motion", length_m=1.45, offset_y=0.0)
    cv2.drawMarker(canvas, (int(cx - center[1] * scale), int(bottom - center[0] * scale)),
                   (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.rectangle(canvas, (0, 0), (width_px, 132), (10, 12, 16), -1)
    title = (f"LiDAR_TOP BEV | {target['class']} track {target['track_id']} | "
             f"{target['timestamp']}")
    cv2.putText(canvas, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (245, 245, 245), 2, cv2.LINE_AA)
    details = (f"raw yaw={target['yaw']:.4f} rad; raw world={target['raw_world_deg']:.2f} deg  |  "
               f"resolved={float(audit['heading_world_deg']):.2f} deg  |  "
               f"motion={float(audit['motion_heading_world_deg']):.2f} deg")
    cv2.putText(canvas, details, (24, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                (215, 220, 225), 1, cv2.LINE_AA)
    details2 = (f"speed={float(audit['world_speed_mps']):.2f} m/s  |  "
                f"axis error={float(audit['motion_axis_error_mod180_deg']):.2f} deg  |  "
                f"confidence={float(audit['heading_confidence']):.3f}  |  "
                f"changed by {target['flip_deg']:.0f} deg")
    cv2.putText(canvas, details2, (24, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                (215, 220, 225), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Orange=raw yaw   Green=resolved heading   Cyan=track motion   Yellow=target 3D box",
                (70, height_px - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.49,
                (230, 230, 230), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def _render_cameras(scene_dir, sample, target_world_corners, center_world,
                    raw_world_deg, heading_world_deg, output_path):
    ann = scene_dir / "annotation"
    sensors = {x["token"]: x for x in json.loads((ann / "sensor.json").read_text(encoding="utf-8"))}
    calibrated = {x["token"]: x for x in json.loads((ann / "calibrated_sensor.json").read_text(encoding="utf-8"))}
    poses = {x["token"]: x for x in json.loads((ann / "ego_pose.json").read_text(encoding="utf-8"))}
    sample_data = {x["token"]: x for x in json.loads((ann / "sample_data.json").read_text(encoding="utf-8"))}
    target_data = []
    for channel, camera in sample["cameras"].items():
        sd = sample_data[camera["sample_data_token"]]
        sensor = sensors[calibrated[sd["calibrated_sensor_token"]]["sensor_token"]]
        cal = calibrated[sd["calibrated_sensor_token"]]
        ep = poses[sd["ego_pose_token"]]
        R_we = _quat_rotation(ep["rotation"])
        t_we = np.asarray(ep["translation"], dtype=np.float64)
        R_ec = _quat_rotation(cal["rotation"])
        t_ec = np.asarray(cal["translation"], dtype=np.float64)
        R_wc = R_we @ R_ec
        t_wc = t_we + R_we @ t_ec
        K = np.asarray(cal["camera_intrinsic"], dtype=np.float64)

        def project(world_points):
            camera_points = (np.asarray(world_points) - t_wc) @ R_wc
            return [(_pixel(p, K), float(p[2])) for p in camera_points]

        projected_corners = project(target_world_corners)
        raw_angle = math.radians(raw_world_deg)
        heading_angle = math.radians(heading_world_deg)
        arrow_world = np.asarray([
            center_world,
            center_world + [3.0 * math.cos(raw_angle), 3.0 * math.sin(raw_angle), 0.0],
            center_world + [3.0 * math.cos(heading_angle), 3.0 * math.sin(heading_angle), 0.0],
        ])
        projected_arrows = project(arrow_world)
        image = cv2.imread(str(scene_dir / camera["filename"]))
        if image is None:
            continue
        visible = False
        edges = ((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6),
                 (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7))
        for a, b in edges:
            pa, za = projected_corners[a]
            pb, zb = projected_corners[b]
            if pa is None or pb is None or za <= 0.1 or zb <= 0.1:
                continue
            if (0 <= pa[0] < image.shape[1] and 0 <= pa[1] < image.shape[0]) or \
                    (0 <= pb[0] < image.shape[1] and 0 <= pb[1] < image.shape[0]):
                visible = True
                cv2.line(image, pa, pb, (0, 240, 255), 2, cv2.LINE_AA)
        pcenter, _ = projected_arrows[0]
        praw, _ = projected_arrows[1]
        phead, _ = projected_arrows[2]
        for end, color in ((praw, (0, 125, 255)), (phead, (80, 245, 80))):
            if pcenter and end:
                cv2.arrowedLine(image, pcenter, end, color, 2, cv2.LINE_AA, tipLength=0.18)
        thumbnail = cv2.resize(image, (384, 216), interpolation=cv2.INTER_AREA)
        tile = np.zeros((258, 384, 3), np.uint8)
        tile[:216] = thumbnail
        name = Path(camera["source_path"]).name
        label = f"{channel} | {name}"
        cv2.putText(tile, label, (5, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(tile, "box visible" if visible else "box projection outside view",
                    (5, 253), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                    (100, 255, 100) if visible else (150, 150, 150), 1, cv2.LINE_AA)
        target_data.append((channel, tile, visible, camera["source_path"]))

    sheet = np.zeros((80 + 2 * 258, 4 * 384, 3), np.uint8)
    cv2.putText(sheet, "Synchronized undistorted camera views | yellow=3D box, orange=raw, green=resolved",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(sheet, "Inspect visible projected box to compare its front direction with the arrows",
                (12, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (195, 205, 215), 1, cv2.LINE_AA)
    for idx, (_, tile, _, _) in enumerate(target_data):
        row, col = divmod(idx, 4)
        y, x = 80 + row * 258, col * 384
        sheet[y:y + 258, x:x + 384] = tile
    cv2.imwrite(str(output_path), sheet)
    return [{"channel": channel, "source_path": source, "projected_box_visible": visible}
            for channel, _, visible, source in target_data]


def visualize_heading_example(scene_dir: Path, timestamp: str, track_id: str,
                              class_name: str, output_dir: Path) -> dict:
    scene_dir, output_dir = scene_dir.resolve(), output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((scene_dir / "run_manifest.json").read_text(encoding="utf-8"))
    audit_path = scene_dir / "annotation" / "heading_audit" / "resolved_world_headings.csv"
    with audit_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    audit = next(r for r in rows if r["timestamp"] == timestamp and
                 r["trackid"] == str(track_id) and r["class"] == class_name)
    if audit["heading_source"] != "world_motion_assumed_forward" or audit["heading_valid"] != "True":
        raise ValueError("Select an object with a direct, motion-resolved heading")
    sample = next(s for s in manifest["samples"]
                  if str(s["reference_label_path"]).endswith(timestamp + ".txt"))
    label_path = Path(sample["reference_label_path"])
    target = None
    boxes = []
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        f = line.split()
        if len(f) < 11 or f[7] not in {"Car", "Rider", "Pedestrian"}:
            continue
        values = list(map(float, f[:7]))
        row = (*values, f[7], f[10])
        boxes.append(row)
        if f[10] == str(track_id) and f[7] == class_name:
            target = {"x": values[0], "y": values[1], "z": values[2],
                      "length": values[3], "width": values[4], "height": values[5],
                      "yaw": values[6], "class": f[7], "track_id": f[10],
                      "timestamp": timestamp}
    if target is None:
        raise ValueError(f"No {class_name} track {track_id} in {label_path}")
    ego_yaw = math.radians(90.0 - float(audit["nav_heading_deg"]))
    target["raw_world_deg"] = math.degrees(ego_yaw + target["yaw"]) % 360.0
    target["flip_deg"] = abs((float(audit["heading_world_deg"]) -
                              target["raw_world_deg"] + 180.0) % 360.0 - 180.0)
    lidar_path = Path(sample["lidar"]["source_path"])
    lidar_png = output_dir / f"heading_{class_name.lower()}_{track_id}_{timestamp}_lidar.png"
    camera_png = output_dir / f"heading_{class_name.lower()}_{track_id}_{timestamp}_cameras.png"
    _render_lidar(lidar_path, boxes, target, audit, lidar_png)

    ann = scene_dir / "annotation"
    lidar_sd_token = sample["lidar"]["sample_data_token"]
    sd_rows = json.loads((ann / "sample_data.json").read_text(encoding="utf-8"))
    ep_rows = {p["token"]: p for p in json.loads((ann / "ego_pose.json").read_text(encoding="utf-8"))}
    lidar_sd = next(sd for sd in sd_rows if sd["token"] == lidar_sd_token)
    anchor_pose = ep_rows[lidar_sd["ego_pose_token"]]
    R_we = _quat_rotation(anchor_pose["rotation"])
    t_we = np.asarray(anchor_pose["translation"], dtype=np.float64)
    corners_ego = _box_corners(target["x"], target["y"], target["z"],
                               target["length"], target["width"], target["height"],
                               target["yaw"])
    corners_world = corners_ego @ R_we.T + t_we
    center_ego = np.asarray([target["x"], target["y"], target["z"]])
    center_world = center_ego @ R_we.T + t_we
    cameras = _render_cameras(scene_dir, sample, corners_world, center_world,
                              target["raw_world_deg"], float(audit["heading_world_deg"]),
                              camera_png)
    return {
        "timestamp": timestamp,
        "class": class_name,
        "track_id": str(track_id),
        "raw_yaw_rad": target["yaw"],
        "raw_world_heading_deg": target["raw_world_deg"],
        "resolved_world_heading_deg": float(audit["heading_world_deg"]),
        "motion_heading_world_deg": float(audit["motion_heading_world_deg"]),
        "speed_mps": float(audit["world_speed_mps"]),
        "axis_motion_error_deg": float(audit["motion_axis_error_mod180_deg"]),
        "confidence": float(audit["heading_confidence"]),
        "flip_deg": target["flip_deg"],
        "lidar_source": str(lidar_path),
        "camera_sources": cameras,
        "lidar_visualization": str(lidar_png),
        "camera_contact_sheet": str(camera_png),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--class-name", choices=("Car", "Rider", "Pedestrian"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = visualize_heading_example(args.scene_dir, args.timestamp, args.track_id,
                                       args.class_name, args.out_dir)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
