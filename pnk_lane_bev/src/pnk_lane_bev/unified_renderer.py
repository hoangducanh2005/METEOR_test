"""Required per-camera artifacts and 6-camera plus large fused-BEV video."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .constants import (
    BEV_DRIVABLE_IDS,
    BEV_LANE_ID,
    CAMERAS,
    CAMERA_LABELS,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEFT_WIDTH,
)
from .io import atomic_json, load_task1_manifest
from .project import bev_stem


def _palette() -> np.ndarray:
    palette = np.zeros((65, 3), dtype=np.uint8)
    for index in range(65):
        palette[index] = ((37 * index + 43) % 255, (83 * index + 97) % 255, (151 * index + 29) % 255)
    palette[7] = (200, 120, 20)
    palette[13] = (70, 70, 70)
    palette[14] = (95, 95, 95)
    palette[23] = (0, 165, 255)
    palette[24] = (0, 255, 255)
    return palette


def _semantic_color(semantic: np.ndarray) -> np.ndarray:
    colors = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    valid = semantic < 65
    colors[valid] = _palette()[semantic[valid]]
    return colors


def _binary_visual(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask.astype(bool)] = color
    return image


def _title(image: np.ndarray, text: str) -> np.ndarray:
    value = image.copy()
    cv2.rectangle(value, (0, 0), (value.shape[1], 34), (18, 18, 18), -1)
    cv2.putText(value, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
    return value


def _read_camera_metrics(path: Path) -> dict[str, dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["camera"]: row for row in csv.DictReader(handle)}


def _bev_visual(data: dict, reference: np.ndarray | None = None) -> np.ndarray:
    drive = data["drivable_mask"].astype(bool)
    lane = data["lane_mask"].astype(bool)
    coverage = data["coverage"].astype(np.float32)
    image = np.zeros((*drive.shape, 3), dtype=np.uint8)
    shade = np.uint8(np.clip(coverage / max(len(CAMERAS), 1), 0, 1) * 50)
    image[..., 0] = shade
    image[..., 1] = shade
    image[..., 2] = shade
    image[drive] = (45, 155, 55)
    image[lane] = (0, 245, 255)
    if reference is not None:
        reference_drive = np.isin(reference, BEV_DRIVABLE_IDS).astype(np.uint8)
        edge = cv2.morphologyEx(reference_drive, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        image[edge] = (220, 125, 35)
        reference_lane = reference == BEV_LANE_ID
        image[reference_lane & ~lane] = (210, 70, 175)
    for row in range(0, image.shape[0], 50):
        cv2.line(image, (0, row), (image.shape[1] - 1, row), (70, 70, 70), 1)
    for col in range(0, image.shape[1], 50):
        cv2.line(image, (col, 0), (col, image.shape[0] - 1), (70, 70, 70), 1)
    ego = (image.shape[1] // 2, image.shape[0] // 2)
    cv2.arrowedLine(image, ego, (ego[0], ego[1] - 38), (255, 255, 255), 3, tipLength=0.25)
    cv2.circle(image, ego, 7, (255, 255, 255), -1)
    return image


def _save_camera_artifacts(
    root: Path,
    camera: str,
    stem: str,
    rgb: np.ndarray,
    arrays: dict,
    camera_bev: np.ndarray,
) -> None:
    target = root / camera
    target.mkdir(parents=True, exist_ok=True)
    semantic = _semantic_color(arrays["semantic"])
    drivable = _binary_visual(arrays["drivable_mask"], (45, 155, 55))
    lane_raw = _binary_visual(arrays["lane_mask_raw"], (0, 165, 255))
    lane_post = _binary_visual(arrays["lane_mask_post"], (0, 245, 255))
    cv2.imwrite(str(target / f"{stem}_rgb.jpg"), rgb)
    cv2.imwrite(str(target / f"{stem}_semantic.png"), semantic)
    cv2.imwrite(str(target / f"{stem}_drivable.png"), drivable)
    cv2.imwrite(str(target / f"{stem}_lane_raw.png"), lane_raw)
    cv2.imwrite(str(target / f"{stem}_lane_post.png"), lane_post)
    cv2.imwrite(str(target / f"{stem}_bev.png"), camera_bev)
    tiles = [
        _title(rgb, "RGB"), _title(semantic, "SEMANTIC MAP"), _title(drivable, "DRIVABLE MASK"),
        _title(lane_raw, "LANE MASK"), _title(lane_post, "LANE POST-PROCESS"),
        _title(cv2.resize(camera_bev, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST), "PER-CAMERA BEV"),
    ]
    debug = np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:])))
    cv2.imwrite(str(target / f"{stem}_debug.jpg"), debug)


def render(
    scene_dir: Path,
    output_dir: Path,
    fps: float = 5.0,
    limit: int | None = None,
) -> dict:
    manifest = load_task1_manifest(scene_dir)
    frames = manifest["frames"][:limit] if limit is not None else manifest["frames"]
    camera_metrics = _read_camera_metrics(output_dir / "metrics" / "camera_summary.csv")
    deliverable = output_dir / "deliverables" / "task2_mask2former_unified_6cam_bev.mp4"
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(deliverable), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {deliverable}")
    layout = (
        "CAM_FRONT_LEFT", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK_WIDE", "CAM_BACK_RIGHT",
    )
    artifacts_root = output_dir / "artifacts" / "per_camera"
    fused_root = output_dir / "artifacts" / "fused_bev"
    frame_root = output_dir / "deliverables" / "frames"
    fused_root.mkdir(parents=True, exist_ok=True)
    frame_root.mkdir(parents=True, exist_ok=True)
    for frame_index, frame in enumerate(frames, 1):
        stem = bev_stem(frame)
        with np.load(output_dir / "bev" / f"{stem}.npz") as npz:
            bev_data = {key: npz[key] for key in npz.files}
        reference_bev = cv2.imread(str(scene_dir / frame["gt_vec"]), cv2.IMREAD_GRAYSCALE)
        fused = _bev_visual(bev_data, reference_bev)
        cv2.imwrite(str(fused_root / f"{stem}.png"), fused)
        canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 18, dtype=np.uint8)
        cell_width = LEFT_WIDTH // 3
        cell_height = 420
        for position, camera in enumerate(layout):
            rgb = cv2.imread(str(scene_dir / frame["imgs"][camera]), cv2.IMREAD_COLOR)
            prediction_path = (
                output_dir / "cache" / "mask2former_unified" / camera /
                f"{int(frame['frame']):04d}.npz"
            )
            with np.load(prediction_path) as npz:
                arrays = {key: npz[key] for key in npz.files}
            camera_bev_path = output_dir / "bev_visual" / "per_camera" / camera / f"{stem}.png"
            camera_bev = cv2.imread(str(camera_bev_path), cv2.IMREAD_COLOR)
            _save_camera_artifacts(artifacts_root, camera, stem, rgb, arrays, camera_bev)
            overlay = rgb.copy()
            drive = arrays["drivable_mask"].astype(bool)
            lane_raw = arrays["lane_mask_raw"].astype(bool)
            lane_post = arrays["lane_mask_post"].astype(bool)
            color = np.zeros_like(overlay)
            color[drive] = (45, 165, 55)
            overlay = cv2.addWeighted(overlay, 1.0, color, 0.40, 0)
            overlay[lane_raw] = (0, 145, 255)
            overlay[lane_post] = (0, 255, 255)
            metric = camera_metrics[camera]
            header = np.zeros((64, IMAGE_WIDTH, 3), dtype=np.uint8)
            cv2.putText(header, CAMERA_LABELS[camera], (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
            line = (
                f"Drive IoU {float(metric['drivable_iou']):.3f}  "
                f"Lane F1 {float(metric['lane_f1']):.3f}  Empty {float(metric['empty_lane_rate']) * 100:.1f}%"
            )
            cv2.putText(header, line, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (215, 225, 230), 1, cv2.LINE_AA)
            panel = cv2.resize(np.vstack((header, overlay)), (cell_width, cell_height), interpolation=cv2.INTER_AREA)
            row, col = divmod(position, 3)
            x0, y0 = col * cell_width, row * cell_height
            canvas[y0:y0 + cell_height, x0:x0 + cell_width] = panel

        header = np.full((120, CANVAS_WIDTH - LEFT_WIDTH, 3), 24, dtype=np.uint8)
        cv2.putText(header, "MASK2FORMER UNIFIED - FUSED BEV", (28, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(header, "green: drivable | yellow: lane | magenta/blue: proxy", (28, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 220, 225), 1, cv2.LINE_AA)
        cv2.putText(header, "Mapillary Swin-L -> class filter -> post-process -> IPM", (28, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (115, 225, 255), 1, cv2.LINE_AA)
        canvas[:120, LEFT_WIDTH:] = header
        resized_bev = cv2.resize(fused, (800, 1280), interpolation=cv2.INTER_NEAREST)
        canvas[135:1415, LEFT_WIDTH + 50:LEFT_WIDTH + 850] = resized_bev
        footer_y = 840
        canvas[footer_y:, :LEFT_WIDTH] = (24, 24, 24)
        lines = (
            f"TASK 2 | frame {int(frame['frame']):04d}/{len(frames) - 1:04d} | timestamp {frame['timestamp']}",
            "ONE MODEL: Mask2Former Swin-L Mapillary Vistas Panoptic",
            "OUTPUT: semantic + drivable + lane | UFLD-v2 absent from inference path",
            "IPM: calibrated ground plane | FUSION: camera x confidence x distance weight",
            "METRICS: source-coupled proxy; human GT is still required for accuracy",
        )
        colors = ((250, 250, 250), (100, 225, 255), (80, 225, 125), (215, 215, 215), (70, 185, 255))
        for line_index, (line, color) in enumerate(zip(lines, colors)):
            cv2.putText(canvas, line, (35, footer_y + 60 + line_index * 55), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2 if line_index < 3 else 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (35, footer_y + 360), (1610, footer_y + 535), (38, 38, 38), -1)
        cv2.putText(canvas, "REQUIRED ARTIFACTS SAVED FOR EVERY CAMERA", (60, footer_y + 405), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "RGB | semantic | drivable | lane raw | lane post | camera BEV", (60, footer_y + 455), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (215, 215, 215), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Fused BEV occupies the complete right column", (60, footer_y + 505), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (215, 215, 215), 1, cv2.LINE_AA)
        frame_path = frame_root / f"{stem}.jpg"
        cv2.imwrite(str(frame_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        writer.write(canvas)
        print(f"[mask2former-render] {frame_index}/{len(frames)} {frame_path.name}", flush=True)
    writer.release()
    summary = {
        "frames": len(frames), "camera_artifact_sets": len(frames) * len(CAMERAS),
        "artifacts_per_camera_frame": ["rgb", "semantic", "drivable", "lane_raw", "lane_post", "bev", "debug"],
        "fused_bev_frames": len(frames), "video": str(deliverable.resolve()), "fps": fps,
        "canvas_wh": [CANVAS_WIDTH, CANVAS_HEIGHT],
    }
    atomic_json(output_dir / "render_summary.json", summary)
    return summary
