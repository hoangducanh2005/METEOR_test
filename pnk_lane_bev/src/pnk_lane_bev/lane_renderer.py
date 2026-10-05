"""Visual evidence for six cameras, per-camera contributions and fused BEV."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .constants import BEV_DRIVABLE_IDS, BEV_LANE_ID, CAMERAS, CAMERA_LABELS
from .io import atomic_json, load_task1_manifest, read_json
from .lane_pipeline import DETECTORS, lane_raster
from .project import bev_stem


CANVAS_WH = (2560, 1440)
LEFT_WIDTH = 1660


def _metrics(path: Path) -> tuple[dict, dict]:
    image_rows, bev_rows = {}, {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            detector = row["detector"]
            frame = int(row["frame"])
            if row["camera"] == "FUSED_BEV":
                bev_rows[(detector, frame)] = row
            else:
                image_rows[(detector, frame, row["camera"])] = row
    return image_rows, bev_rows


def _camera_panel(image: np.ndarray, record: dict, metric: dict, title: str) -> np.ndarray:
    canvas = image.copy()
    for index, lane in enumerate(record["lanes"]):
        points = np.rint(np.asarray(lane["points"], np.float32)).astype(np.int32)
        color = ((37 + 73 * index) % 255, 255, (220 - 37 * index) % 255)
        if len(points) >= 2:
            cv2.polylines(canvas, [points[:, None, :]], False, color, 4, cv2.LINE_AA)
    header = np.full((74, image.shape[1], 3), 24, np.uint8)
    cv2.putText(header, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (250, 250, 250), 2, cv2.LINE_AA)
    text = (
        f"F1 {float(metric['lane_f1']):.3f} [PROXY] | lanes {record['lane_count']} | "
        f"conf {record['mean_confidence']:.2f} | {record['runtime_ms']:.1f} ms"
    )
    cv2.putText(header, text, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (205, 220, 225), 1, cv2.LINE_AA)
    if not record["valid"]:
        cv2.putText(canvas, "NO VALID LANE AFTER FILTER", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 80, 255), 2)
    return np.vstack((header, canvas))


def _bev_panel(payload: dict, proxy: np.ndarray, metric: dict, detector: str) -> np.ndarray:
    drive = payload["drivable_mask"].astype(bool)
    lane = payload["lane_mask"].astype(bool)
    canvas = np.full((*drive.shape, 3), 20, np.uint8)
    canvas[drive] = (45, 135, 55)
    canvas[lane] = (0, 245, 255)
    proxy_lane = proxy == BEV_LANE_ID
    canvas[proxy_lane & ~lane] = (220, 60, 180)
    proxy_drive = np.isin(proxy, BEV_DRIVABLE_IDS).astype(np.uint8)
    boundary = cv2.morphologyEx(proxy_drive, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    canvas[boundary] = (220, 120, 40)
    ego = (canvas.shape[1] // 2, canvas.shape[0] // 2)
    cv2.arrowedLine(canvas, ego, (ego[0], ego[1] - 40), (255, 255, 255), 3, tipLength=0.25)
    cv2.circle(canvas, ego, 7, (255, 255, 255), -1)
    header = np.full((100, canvas.shape[1], 3), 24, np.uint8)
    cv2.putText(header, f"FUSED LANE BEV | {detector}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (250, 250, 250), 2)
    chamfer = metric.get("bev_chamfer_m", "")
    chamfer_text = f"{float(chamfer):.3f} m" if chamfer not in ("", None) else "N/A"
    cv2.putText(
        header,
        f"F1 {float(metric['lane_f1']):.3f} | Chamfer {chamfer_text} [PROXY]",
        (16, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (80, 230, 255),
        1,
    )
    cv2.putText(header, "cyan prediction | magenta missed proxy | green drivable", (16, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1)
    return np.vstack((header, canvas))


def render(
    scene_dir: Path,
    output_dir: Path,
    detectors: tuple[str, ...] = DETECTORS,
    fps: float = 5.0,
    limit: int | None = None,
) -> dict:
    manifest = load_task1_manifest(scene_dir)
    frames = manifest["frames"][:limit] if limit is not None else manifest["frames"]
    image_metrics, bev_metrics = _metrics(output_dir / "lane_v2" / "metrics" / "frame_camera_metrics.csv")
    # BEV rows are stored separately.
    _, loaded_bev_metrics = _metrics(output_dir / "lane_v2" / "metrics" / "frame_bev_metrics.csv")
    bev_metrics.update(loaded_bev_metrics)
    render_summary_path = output_dir / "lane_v2" / "render_summary.json"
    summaries = read_json(render_summary_path) if render_summary_path.exists() else {}
    layout = (
        "CAM_FRONT_LEFT", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK_WIDE", "CAM_BACK_RIGHT",
    )
    for detector in detectors:
        deliverable = output_dir / "lane_v2" / detector / "deliverables"
        frames_dir = deliverable / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        video = deliverable / f"task2_{detector}_6cam_bev.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, CANVAS_WH)
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {video}")
        for frame_index, frame in enumerate(frames, 1):
            canvas = np.full((CANVAS_WH[1], CANVAS_WH[0], 3), 18, np.uint8)
            cell_w, cell_h = LEFT_WIDTH // 3, 420
            for position, camera in enumerate(layout):
                image = cv2.imread(str(scene_dir / frame["imgs"][camera]), cv2.IMREAD_COLOR)
                record = read_json(
                    output_dir / "lane_v2" / detector / "cache" / camera / f"{int(frame['frame']):04d}.json"
                )
                metric = image_metrics[(detector, int(frame["frame"]), camera)]
                panel = _camera_panel(image, record, metric, CAMERA_LABELS[camera])
                panel = cv2.resize(panel, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                row, col = divmod(position, 3)
                canvas[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w] = panel
            stem = bev_stem(frame)
            with np.load(output_dir / "lane_v2" / detector / "bev" / f"{stem}.npz") as data:
                payload = {key: data[key] for key in data.files}
            proxy = cv2.imread(str(scene_dir / frame["gt_vec"]), cv2.IMREAD_GRAYSCALE)
            bev = _bev_panel(payload, proxy, bev_metrics[(detector, int(frame["frame"]))], detector)
            bev = cv2.resize(bev, (650, 1170), interpolation=cv2.INTER_NEAREST)
            canvas[20:1190, LEFT_WIDTH + 15 : LEFT_WIDTH + 665] = bev

            # Six compact contribution maps make camera dominance inspectable.
            for camera_index, camera in enumerate(CAMERAS):
                contribution = payload["per_camera_contribution_share"][camera_index].astype(np.float32)
                heat = cv2.applyColorMap(np.uint8(np.clip(contribution, 0, 1) * 255), cv2.COLORMAP_TURBO)
                heat = cv2.resize(heat, (105, 168), interpolation=cv2.INTER_NEAREST)
                row, col = divmod(camera_index, 2)
                x0 = LEFT_WIDTH + 675 + col * 110
                y0 = 90 + row * 205
                canvas[y0 : y0 + 168, x0 : x0 + 105] = heat
                cv2.putText(canvas, CAMERA_LABELS[camera], (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (230, 230, 230), 1)
            cv2.putText(canvas, "CAMERA CONTRIBUTION", (LEFT_WIDTH + 672, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (245, 245, 245), 1)

            footer = 860
            cv2.putText(canvas, f"LANE BRANCH V2 | frame {int(frame['frame']):04d} | {detector}", (35, footer + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (250, 250, 250), 2)
            cv2.putText(canvas, "Raw -> Calib_2 undistortion -> camera ROI -> detector -> threshold -> IPM", (35, footer + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (210, 220, 225), 2)
            cv2.putText(canvas, "Fusion weight = camera prior x confidence x distance reliability", (35, footer + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (80, 230, 255), 2)
            cv2.putText(canvas, "Per-camera weights and thresholds: configs/lane/cameras/pnk_6cam.yaml", (35, footer + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (210, 210, 210), 1)
            cv2.putText(canvas, "Reference: Task 1 pseudo-GT | status: PASS_PROXY_ONLY", (35, footer + 250), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 190, 255), 2)
            cv2.putText(canvas, "Preprocess triptychs and full contribution maps are stored beside this video.", (35, footer + 300), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 190, 190), 1)
            frame_path = frames_dir / f"{stem}.jpg"
            cv2.imwrite(str(frame_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
            writer.write(canvas)
            print(f"[lane-v2-render:{detector}] {frame_index}/{len(frames)}", flush=True)
        writer.release()
        summaries[detector] = {
            "frames": len(frames),
            "video": str(video.resolve()),
            "canvas_wh": list(CANVAS_WH),
            "preprocess_debug": str((output_dir / "lane_v2" / "debug" / "preprocess").resolve()),
            "contribution_maps": str((output_dir / "lane_v2" / detector / "bev_visual" / "contribution").resolve()),
        }
    atomic_json(render_summary_path, summaries)
    return summaries
