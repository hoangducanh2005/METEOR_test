"""Task 2 evidence renderer: six cameras plus a large right-side BEV."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .constants import (
    BEV_DRIVABLE_IDS,
    BEV_LANE_ID,
    BEV_DISPLAY_WIDTH,
    CAMERAS,
    CAMERA_LABELS,
    CAMERA_GRID_HEIGHT,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEFT_WIDTH,
    SEG2D_LANE_ID,
    SEG2D_ROAD_ID,
)
from .io import atomic_json, load_task1_manifest, read_json
from .project import bev_stem, lane_raster


def _read_metrics(path: Path) -> tuple[dict, dict]:
    images, bev = {}, {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if row["record_type"] == "image":
                images[(frame, row["camera"])] = row
            else:
                bev[frame] = row
    return images, bev


def _score(row: dict, key: str, suffix: str = "") -> str:
    value = row.get(key, "")
    return f"{float(value):.3f}{suffix}" if value not in (None, "", "None") else "N/A"


def _overlay_camera(
    image: np.ndarray,
    drive_mask: np.ndarray,
    lane_mask: np.ndarray,
    reference: np.ndarray,
    title: str,
    metric: dict,
) -> tuple[np.ndarray, np.ndarray]:
    view = image.copy()
    color = np.zeros_like(view)
    color[drive_mask] = (50, 190, 55)
    view = cv2.addWeighted(view, 1.0, color, 0.42, 0)
    lane_pixels = lane_mask.astype(bool)
    view[lane_pixels] = (0, 245, 255)
    evidence = image.copy()
    road_ref = reference == SEG2D_ROAD_ID
    tp, fp, fn = drive_mask & road_ref, drive_mask & ~road_ref, ~drive_mask & road_ref
    evidence[tp] = (30, 190, 30)
    evidence[fp] = (30, 30, 230)
    evidence[fn] = (230, 90, 30)
    evidence[lane_pixels] = (0, 255, 255)

    header = np.zeros((64, IMAGE_WIDTH, 3), np.uint8)
    cv2.putText(header, title, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    info = f"Road IoU {_score(metric, 'iou')} [PROXY]  Lane F1 {_score(metric, 'lane_f1')} [PROXY]  lanes {metric.get('lane_count', '0')}"
    cv2.putText(header, info, (15, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 220, 225), 1, cv2.LINE_AA)
    return np.vstack((header, view)), evidence


def _bev_visual(data: dict, proxy: np.ndarray) -> np.ndarray:
    drive = data["drivable_mask"].astype(bool)
    lane = data["lane_mask"].astype(bool)
    coverage = data["coverage"].astype(np.float32)
    canvas = np.zeros((*drive.shape, 3), np.uint8)
    shade = np.uint8(np.clip(coverage / 6.0, 0, 1) * 55)
    canvas[..., 0] = shade
    canvas[..., 1] = shade
    canvas[..., 2] = shade
    canvas[drive] = (52, 145, 62)
    canvas[lane] = (0, 240, 255)
    proxy_drive = np.isin(proxy, BEV_DRIVABLE_IDS).astype(np.uint8)
    proxy_edge = cv2.morphologyEx(proxy_drive, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    canvas[proxy_edge] = (210, 120, 30)
    proxy_lane = proxy == BEV_LANE_ID
    canvas[proxy_lane & ~lane] = (205, 70, 170)
    for row in range(0, canvas.shape[0], 50):
        cv2.line(canvas, (0, row), (canvas.shape[1] - 1, row), (75, 75, 75), 1)
    for col in range(0, canvas.shape[1], 50):
        cv2.line(canvas, (col, 0), (col, canvas.shape[0] - 1), (75, 75, 75), 1)
    ego = (canvas.shape[1] // 2, canvas.shape[0] // 2)
    cv2.arrowedLine(canvas, ego, (ego[0], ego[1] - 35), (255, 255, 255), 3, tipLength=0.25)
    cv2.circle(canvas, ego, 7, (255, 255, 255), -1)
    return canvas


def render(scene_dir: Path, output_dir: Path, fps: float = 5.0, limit: int | None = None) -> dict:
    manifest = load_task1_manifest(scene_dir)
    frames = manifest["frames"][:limit] if limit is not None else manifest["frames"]
    image_metrics, bev_metrics = _read_metrics(output_dir / "metrics" / "frame_metrics.csv")
    video_path = output_dir / "task2_6cam_bev.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (CANVAS_WIDTH, CANVAS_HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot open MP4 writer: {video_path}")
    layout = (
        "CAM_FRONT_LEFT", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK_WIDE", "CAM_BACK_RIGHT",
    )
    worst = []
    for index, frame in enumerate(frames, 1):
        canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 18, np.uint8)
        with np.load(scene_dir / frame["seg2d21"]) as data:
            proxy_semantic = data["seg"]
        camera_order = list(manifest["cams"])
        cell_width, cell_height = LEFT_WIDTH // 3, CAMERA_GRID_HEIGHT // 2
        for position, camera in enumerate(layout):
            image = cv2.imread(str(scene_dir / frame["imgs"][camera]), cv2.IMREAD_COLOR)
            with np.load(output_dir / "cache" / "mask2former_small" / camera / f"{int(frame['frame']):04d}.npz") as data:
                drive = data["mask"].astype(bool)
            lanes = lane_raster(read_json(output_dir / "cache" / "ufld_v2" / camera / f"{int(frame['frame']):04d}.json")) > 0.2
            reference = cv2.resize(proxy_semantic[camera_order.index(camera)], (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
            panel, evidence = _overlay_camera(image, drive, lanes, reference, CAMERA_LABELS[camera], image_metrics[(int(frame["frame"]), camera)])
            panel = cv2.resize(panel, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
            row, col = divmod(position, 3)
            x0, y0 = col * cell_width, row * cell_height
            canvas[y0 : y0 + cell_height, x0 : x0 + cell_width] = panel
            evidence_path = output_dir / "evidence" / "image_space" / camera / f"{bev_stem(frame)}.png"
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(evidence_path), evidence)
            worst.append((float(image_metrics[(int(frame["frame"]), camera)]["iou"]), evidence_path, camera, int(frame["frame"])))

        stem = bev_stem(frame)
        with np.load(output_dir / "bev" / f"{stem}.npz") as npz:
            bev_data = {key: npz[key] for key in npz.files}
        proxy_bev = cv2.imread(str(scene_dir / frame["gt_vec"]), cv2.IMREAD_GRAYSCALE)
        bev_view = _bev_visual(bev_data, proxy_bev)
        evidence_path = output_dir / "evidence" / "bev" / f"{stem}.png"
        cv2.imwrite(str(evidence_path), bev_view)
        metric = bev_metrics[int(frame["frame"])]
        header = np.full((120, BEV_DISPLAY_WIDTH, 3), 25, np.uint8)
        cv2.putText(header, "FUSED BEV - 6 CAMERA IPM", (32, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        line1 = f"Drive IoU {_score(metric, 'drive_iou')}  Dice {_score(metric, 'drive_dice')} [PROXY]"
        line2 = f"Lane F1 {_score(metric, 'lane_f1')}  Chamfer {_score(metric, 'lane_chamfer_m', ' m')} [PROXY]"
        cv2.putText(header, line1, (32, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (140, 235, 150), 1, cv2.LINE_AA)
        cv2.putText(header, line2, (32, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (90, 230, 255), 1, cv2.LINE_AA)
        canvas[:120, LEFT_WIDTH:] = header
        resized_bev = cv2.resize(bev_view, (800, 1280), interpolation=cv2.INTER_NEAREST)
        canvas[135:1415, LEFT_WIDTH + 50 : LEFT_WIDTH + 850] = resized_bev
        cv2.putText(canvas, "front +80m", (LEFT_WIDTH + 58, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, "rear -80m", (LEFT_WIDTH + 58, 1400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)

        footer_y = CAMERA_GRID_HEIGHT
        canvas[footer_y:, :LEFT_WIDTH] = (24, 24, 24)
        cv2.putText(canvas, f"TASK 2 | frame {int(frame['frame']):04d}/0070 | timestamp {frame['timestamp']}", (35, footer_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (250, 250, 250), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Mask2Former Swin-Small: drivable road segmentation", (35, footer_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (105, 225, 115), 2, cv2.LINE_AA)
        cv2.putText(canvas, "UFLD-v2 ResNet-18: lane curves", (35, footer_y + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 230, 250), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Calibrated IPM + confidence weighted six-camera fusion", (35, footer_y + 185), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (205, 205, 205), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (35, footer_y + 230), (785, footer_y + 405), (38, 38, 38), -1)
        cv2.putText(canvas, "EVIDENCE", (60, footer_y + 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "GREEN drivable | YELLOW lane | BLUE proxy boundary", (60, footer_y + 315), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Reference: Task 1 pseudo-GT | status: PASS_PROXY_ONLY", (60, footer_y + 360), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (80, 190, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Scores are agreement metrics, not human-GT accuracy.", (60, footer_y + 400), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (840, footer_y + 230), (1615, footer_y + 405), (38, 38, 38), -1)
        cv2.putText(canvas, "CURRENT FUSED BEV", (865, footer_y + 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, line1, (865, footer_y + 320), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (140, 235, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (865, footer_y + 365), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (90, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Valid coverage {_score(metric, 'valid_coverage')}", (865, footer_y + 405), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)

        frame_path = output_dir / "frames" / f"{stem}.png"
        cv2.imwrite(str(frame_path), canvas)
        writer.write(canvas)
        print(f"[render] {index}/{len(frames)} {frame_path.name}", flush=True)
    writer.release()

    gallery_items = sorted(worst, key=lambda item: item[0])[:12]
    gallery = np.full((3 * 300, 4 * 480, 3), 22, np.uint8)
    for position, (iou, path, camera, frame_number) in enumerate(gallery_items):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        tile = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
        row, col = divmod(position, 4)
        gallery[row * 300 : row * 300 + 270, col * 480 : (col + 1) * 480] = tile
        cv2.putText(gallery, f"frame {frame_number:04d} {camera} IoU {iou:.3f} [PROXY]", (col * 480 + 8, row * 300 + 292), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_dir / "evidence" / "failure_gallery.png"), gallery)
    summary = {"frames": len(frames), "fps": fps, "video": str(video_path.resolve()), "canvas_wh": [CANVAS_WIDTH, CANVAS_HEIGHT]}
    atomic_json(output_dir / "render_summary.json", summary)
    return summary

