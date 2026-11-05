"""Render a small nuScenes/NavSim camera autolabel preview for the README.

This script runs the repository's Mask2Former Swin-L Mapillary checkpoint on
the six nuScenes camera channels around one requested CAM_FRONT timestamp. It
creates a six-camera panoptic contact sheet and a short raw-vs-overlay video.
It does not claim to run the T4Dataset adapter or METEOR factory on nuScenes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pnk_comet_autolabel.mask2former import DEFAULT_MODEL, Mask2FormerRunner
from pnk_comet_autolabel.taxonomy import map_mapillary_label


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

TIMESTAMP_RE = re.compile(r"__(\d+)\.jpg$")
LIDAR_TIMESTAMP_RE = re.compile(r"__(\d+)\.pcd\.bin$")


def parse_timestamp(path: Path) -> int:
    match = TIMESTAMP_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot read timestamp from nuScenes filename: {path.name}")
    return int(match.group(1))


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def color_for(label: str) -> tuple[int, int, int]:
    key = label.lower().replace(" ", "_")
    palette = {
        "road": (190, 68, 185),
        "sidewalk": (248, 35, 206),
        "lane_marking": (20, 220, 255),
        "car": (38, 90, 255),
        "truck": (255, 130, 15),
        "bus": (235, 70, 40),
        "person": (255, 35, 110),
        "rider": (255, 170, 35),
        "bicycle": (20, 225, 140),
        "motorcycle": (255, 110, 25),
        "traffic_light": (60, 235, 65),
        "traffic_sign": (45, 200, 230),
        "building": (100, 160, 190),
        "vegetation": (50, 185, 55),
        "sky": (75, 125, 210),
        "wall_fence": (155, 155, 155),
    }
    if key in palette:
        return palette[key]
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return tuple(65 + byte % 170 for byte in digest[:3])


def infer_all_segments(runner: Mask2FormerRunner, image_path: Path):
    """Return the raw 65-class panoptic result, including unmapped classes."""
    image = Image.open(image_path).convert("RGB")
    inputs = runner.processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(runner.device)
    if runner.device.type == "cuda":
        pixel_values = pixel_values.half()
    with runner.torch.inference_mode():
        with runner.torch.autocast(
            device_type=runner.device.type, enabled=runner.device.type == "cuda"
        ):
            outputs = runner.model(pixel_values=pixel_values)
    result = runner.processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]
    panoptic = result["segmentation"].detach().cpu().numpy().astype(np.int32)
    labels = runner.model.config.id2label
    segments = []
    for item in result["segments_info"]:
        label_id = int(item["label_id"])
        label = labels.get(label_id, labels.get(str(label_id), str(label_id)))
        segments.append({
            "segment_id": int(item["id"]),
            "source_label_id": label_id,
            "source_label": str(label),
            "target_category": map_mapillary_label(str(label)),
            "score": float(item.get("score", 1.0)),
        })
    return image, panoptic, segments


def overlay_image(image: Image.Image, panoptic: np.ndarray, segments: list[dict]) -> Image.Image:
    rgb = np.asarray(image, dtype=np.uint8).copy()
    tint = np.zeros_like(rgb)
    alpha = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for segment in segments:
        mask = panoptic == segment["segment_id"]
        if not mask.any():
            continue
        label = segment["target_category"] or segment["source_label"]
        tint[mask] = color_for(str(label))
        alpha[mask] = 105
    a = alpha.astype(np.float32)[..., None] / 255.0
    blended = (rgb.astype(np.float32) * (1.0 - a) + tint.astype(np.float32) * a)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def nearest_camera_image(root: Path, scene_prefix: str, channel: str,
                         timestamp_us: int) -> Path:
    pattern = f"{scene_prefix}__{channel}__*.jpg"
    candidates = list((root / "samples" / channel).glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No frames for {channel} matching {pattern}")
    return min(candidates, key=lambda path: abs(parse_timestamp(path) - timestamp_us))


def nearest_lidar_scan(root: Path, scene_prefix: str, timestamp_us: int) -> Path:
    candidates = list((root / "samples" / "LIDAR_TOP").glob(
        f"{scene_prefix}__LIDAR_TOP__*.pcd.bin"))
    if not candidates:
        raise FileNotFoundError(f"No LIDAR_TOP sample for scene {scene_prefix}")
    def stamp(path: Path) -> int:
        match = LIDAR_TIMESTAMP_RE.search(path.name)
        if not match:
            raise ValueError(f"Cannot read LiDAR timestamp: {path.name}")
        return int(match.group(1))
    return min(candidates, key=lambda path: abs(stamp(path) - timestamp_us))


def render_lidar_bev(path: Path, out_path: Path, timestamp_us: int) -> dict:
    """Render raw nuScenes LIDAR_TOP points in sensor-frame top-down view."""
    match = LIDAR_TIMESTAMP_RE.search(path.name)
    scan_us = int(match.group(1))
    points = np.fromfile(path, dtype=np.float32)
    if points.size % 5:
        raise ValueError(f"Unexpected nuScenes LiDAR record size: {path}")
    points = points.reshape(-1, 5)
    x_min, x_max = -15.0, 80.0
    y_min, y_max = -50.0, 50.0
    z_min, z_max = -3.0, 5.0
    valid = (np.isfinite(points[:, :3]).all(axis=1)
             & (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
             & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
             & (points[:, 2] >= z_min) & (points[:, 2] <= z_max))
    xyz = points[valid, :3]
    width, height = 1100, 830
    left, top, plot_w, plot_h = 70, 95, 900, 700
    canvas = np.full((height, width, 3), (15, 22, 30), dtype=np.uint8)
    u = left + ((y_max - xyz[:, 1]) / (y_max - y_min) * plot_w).astype(np.int32)
    v = top + ((x_max - xyz[:, 0]) / (x_max - x_min) * plot_h).astype(np.int32)
    z_norm = np.clip((xyz[:, 2] - z_min) / (z_max - z_min) * 255, 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(z_norm.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    canvas[v, u] = colors
    for forward_m in range(0, 81, 10):
        y = top + int((x_max - forward_m) / (x_max - x_min) * plot_h)
        cv2.line(canvas, (left, y), (left + plot_w, y), (48, 60, 72), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{forward_m}m", (8, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (188, 204, 220), 1, cv2.LINE_AA)
    for lateral_m in range(-40, 41, 10):
        x = left + int((y_max - lateral_m) / (y_max - y_min) * plot_w)
        cv2.line(canvas, (x, top), (x, top + plot_h), (40, 52, 65), 1, cv2.LINE_AA)
    ego_u = left + int((y_max - 0.0) / (y_max - y_min) * plot_w)
    ego_v = top + int((x_max - 0.0) / (x_max - x_min) * plot_h)
    cv2.drawMarker(canvas, (ego_u, ego_v), (255, 255, 255), cv2.MARKER_TRIANGLE_UP,
                   22, 2, cv2.LINE_AA)
    cv2.putText(canvas, "nuScenes LIDAR_TOP | raw sensor-frame BEV", (70, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 249, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{path.name} | dt={(scan_us - timestamp_us) / 1000:+.1f} ms | "
                f"{len(xyz):,} points", (70, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (186, 202, 218), 1, cv2.LINE_AA)
    cv2.putText(canvas, "x forward  |  y left", (left + 6, top + plot_h + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, (186, 202, 218), 1, cv2.LINE_AA)
    cv2.putText(canvas, "height", (1000, 150), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (220, 232, 244), 1, cv2.LINE_AA)
    for i, z in enumerate(np.linspace(z_max, z_min, 9)):
        c = cv2.applyColorMap(np.array([[int((z - z_min) / (z_max - z_min) * 255)]],
                                       dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
        y = 180 + i * 55
        cv2.line(canvas, (1000, y), (1040, y), tuple(int(v) for v in c), 12)
        cv2.putText(canvas, f"{z:+.0f} m", (1048, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (190, 205, 220), 1, cv2.LINE_AA)
    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(out_path, optimize=True)
    return {"file": path.name, "timestamp_us": scan_us,
            "delta_ms": (scan_us - timestamp_us) / 1000.0,
            "points_in_view": int(len(xyz)), "points_total": int(len(points))}


def make_contact_sheet(items: list[dict], out_path: Path, target_us: int) -> None:
    tile_w, tile_h, header_h, gap = 800, 450, 72, 10
    canvas = Image.new("RGB", (tile_w * 3 + gap * 4, header_h + tile_h * 2 + gap * 3),
                       (14, 21, 31))
    draw = ImageDraw.Draw(canvas)
    title = font(30)
    draw.text((gap * 2, 17), "nuScenes / NavSim · Mask2Former Swin-L · Mapillary Vistas",
              fill=(245, 249, 255), font=title)
    label_font = font(21)
    for index, item in enumerate(items):
        row, col = divmod(index, 3)
        x, y = gap + col * (tile_w + gap), header_h + gap + row * (tile_h + gap)
        image = item["overlay"].resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + tile_w, y + 36), fill=(9, 15, 24))
        dt_ms = (item["timestamp_us"] - target_us) / 1000.0
        caption = f"{item['channel']}  |  Δt {dt_ms:+.1f} ms  |  {len(item['segments'])} segments"
        draw.text((x + 10, y + 6), caption, fill=(248, 250, 255), font=label_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)


def make_sequence_video(items: list[dict], out_path: Path, fps: float,
                        target_us: int) -> None:
    width, height, title_h = 1920, 540, 48
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height + title_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {out_path}")
    try:
        for item in items:
            raw = item["image"].resize((width // 2, height), Image.Resampling.LANCZOS)
            overlay = item["overlay"].resize((width // 2, height), Image.Resampling.LANCZOS)
            frame = Image.new("RGB", (width, height + title_h), (13, 19, 28))
            frame.paste(raw, (0, title_h))
            frame.paste(overlay, (width // 2, title_h))
            draw = ImageDraw.Draw(frame)
            dt_s = (item["timestamp_us"] - target_us) / 1_000_000.0
            draw.text((18, 11),
                      f"CAM_FRONT  |  raw  →  panoptic overlay  |  Δt {dt_s:+.1f}s  |  "
                      f"{item['timestamp_us']}", fill=(244, 248, 255), font=font(24))
            bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\nuscenes"))
    parser.add_argument("--start-image", type=Path, default=Path(
        r"D:\nuscenes\samples\CAM_FRONT\n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151353262404.jpg"
    ))
    parser.add_argument("--output", type=Path, default=Path(
        r"navsim_visuals\output\navsim_mask2former_n008_1533151353262404"
    ))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sequence-frames", type=int, default=15)
    parser.add_argument("--sequence-step-ms", type=int, default=200)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    if not args.start_image.is_file():
        raise FileNotFoundError(args.start_image)
    target_us = parse_timestamp(args.start_image)
    scene_prefix = args.start_image.name.split("__CAM_FRONT__", 1)[0]
    cache_dir = args.output / "predictions"
    overlay_dir = args.output / "overlays"
    cache_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    center_items = []
    for channel in CAMERAS:
        path = nearest_camera_image(args.data_root, scene_prefix, channel, target_us)
        center_items.append({"channel": channel, "path": path,
                             "timestamp_us": parse_timestamp(path)})

    n = max(1, args.sequence_frames)
    step_us = args.sequence_step_ms * 1000
    first_offset = -(n // 2) * step_us
    video_paths = [nearest_camera_image(args.data_root, scene_prefix, "CAM_FRONT",
                                        target_us + first_offset + i * step_us)
                   for i in range(n)]

    unique_paths = list(dict.fromkeys([item["path"] for item in center_items] + video_paths))
    pending = []
    for path in unique_paths:
        stem = path.stem
        if not (cache_dir / f"{stem}.npz").exists() or not (cache_dir / f"{stem}.json").exists():
            pending.append(path)

    runner = Mask2FormerRunner(args.model, args.device) if pending else None
    elapsed_inference = 0.0
    for index, path in enumerate(pending, 1):
        start = time.perf_counter()
        image, panoptic, segments = infer_all_segments(runner, path)
        elapsed_inference += time.perf_counter() - start
        stem = path.stem
        np.savez_compressed(cache_dir / f"{stem}.npz", panoptic=panoptic)
        (cache_dir / f"{stem}.json").write_text(json.dumps({
            "filename": str(path), "timestamp_us": parse_timestamp(path),
            "segments": segments, "model": args.model,
        }, indent=2), encoding="utf-8")
        overlay = overlay_image(image, panoptic, segments)
        overlay.save(overlay_dir / f"{stem}_overlay.jpg", quality=92)
        print(f"[{index}/{len(pending)}] {path.name}: {len(segments)} segments, "
              f"{time.perf_counter() - start:.2f}s", flush=True)

    def load_render(path: Path, channel: str | None = None) -> dict:
        stem = path.stem
        with np.load(cache_dir / f"{stem}.npz") as data:
            panoptic = data["panoptic"]
        meta = json.loads((cache_dir / f"{stem}.json").read_text(encoding="utf-8"))
        image = Image.open(path).convert("RGB")
        overlay = overlay_image(image, panoptic, meta["segments"])
        return {"channel": channel or "CAM_FRONT", "path": path,
                "timestamp_us": parse_timestamp(path), "image": image,
                "overlay": overlay, "segments": meta["segments"]}

    center_rendered = [load_render(item["path"], item["channel"]) for item in center_items]
    make_contact_sheet(center_rendered, args.output / "navsim_n008_6cam_panoptic.png", target_us)
    front_center = next(item for item in center_rendered if item["channel"] == "CAM_FRONT")
    front_center["overlay"].save(args.output / "navsim_n008_cam_front_panoptic.png", optimize=True)

    video_rendered = [load_render(path) for path in video_paths]
    make_sequence_video(video_rendered, args.output / "navsim_n008_cam_front_panoptic.mp4",
                        args.fps, target_us)
    lidar_path = nearest_lidar_scan(args.data_root, scene_prefix, target_us)
    lidar_summary = render_lidar_bev(lidar_path,
                                     args.output / "navsim_n008_lidar_bev.png", target_us)

    summary = {
        "source_dataset": "nuScenes data used by the NavSim workflow",
        "scene_prefix": scene_prefix,
        "start_image": str(args.start_image),
        "anchor_timestamp_us": target_us,
        "camera_channels": CAMERAS,
        "camera_images_at_anchor": [{"channel": item["channel"], "file": item["path"].name,
                                      "timestamp_us": item["timestamp_us"],
                                      "delta_ms": (item["timestamp_us"] - target_us) / 1000.0,
                                      "segments": len(item["segments"])}
                                     for item in center_rendered],
        "sequence_frames_requested": n,
        "sequence_frames_rendered": len(video_rendered),
        "model": args.model,
        "device": str(runner.device) if runner else args.device,
        "new_inference_images": len(pending),
        "inference_seconds_excluding_model_load": elapsed_inference,
        "lidar_top": lidar_summary,
        "metrics_note": "No reference masks are evaluated; segment count is not an accuracy metric.",
        "factory_note": "This visualization run is camera-only Mask2Former inference. It does not convert nuScenes to T4Dataset or run METEOR Autolabel Factory.",
        "outputs": ["navsim_n008_6cam_panoptic.png",
                    "navsim_n008_cam_front_panoptic.png",
                    "navsim_n008_cam_front_panoptic.mp4",
                    "navsim_n008_lidar_bev.png"],
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
