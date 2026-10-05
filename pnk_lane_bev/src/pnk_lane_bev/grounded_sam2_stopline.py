"""Grounding DINO -> SAM 2.1 stop-line branch for raw PNK camera frames.

The Grounding DINO text prompt localizes candidate stop lines. SAM 2.1 receives
the candidate box as a video prompt and maintains a streaming inference session
per camera so the mask can propagate when a later frame has no fresh detection.
Mask2Former remains the source for Mapillary lane and drivable classes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2VideoModel,
    Sam2VideoProcessor,
)

from pnk_lane_bev.mask2former_unified import MAPILLARY_ID2LABEL, Mask2FormerDetector
from pnk_lane_bev.unified_config import load_unified_config, resolve_class_ids


CAMERAS = (
    "CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
PROMPT = "stop line."
GDINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM2_ID = "facebook/sam2.1-hiera-small"


def _rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _postprocess_stop(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros_like(binary)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= 20:
            keep[labels == i] = 1
    return keep


def _overlay(rgb: np.ndarray, mask: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    color = np.zeros_like(out)
    color[:, :, 2] = 255
    active = mask.astype(bool)
    out[active] = (0.45 * out[active] + 0.55 * color[active]).astype(np.uint8)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(out, (round(x0), round(y0)), (round(x1), round(y1)), (0, 220, 255), 2)
    return out


class GroundedSam2StopLine:
    def __init__(self, device: str, cache_dir: Path, box_threshold: float, text_threshold: float):
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.gd_processor = AutoProcessor.from_pretrained(GDINO_ID, cache_dir=cache_dir)
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GDINO_ID, cache_dir=cache_dir, torch_dtype=self.dtype
        ).to(self.device).eval()
        self.sam_processor = Sam2VideoProcessor.from_pretrained(SAM2_ID, cache_dir=cache_dir)
        self.sam_model = Sam2VideoModel.from_pretrained(
            SAM2_ID, cache_dir=cache_dir, torch_dtype=self.dtype
        ).to(self.device).eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    @torch.inference_mode()
    def ground(self, rgb: np.ndarray) -> tuple[list[list[float]], list[float]]:
        image = Image.fromarray(rgb)
        inputs = self.gd_processor(images=image, text=PROMPT, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
        outputs = self.gd_model(**inputs)
        result = self.gd_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = result["boxes"].detach().float().cpu().numpy().tolist()
        scores = result["scores"].detach().float().cpu().numpy().tolist()
        return boxes, scores


def run(preflight: Path, output: Path, cache: Path, device: str = "auto",
        limit: int | None = None, box_threshold: float = 0.22,
        text_threshold: float = 0.18, force: bool = False) -> dict:
    manifest = json.loads(preflight.read_text(encoding="utf-8"))
    frames = manifest["samples"][:limit]
    if not frames:
        raise ValueError("Preflight contains no accepted synchronized camera frames")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    output.mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(exist_ok=True)
    for group in ("stop_line", "lane", "drivable", "semantic"):
        (output / "masks" / group).mkdir(exist_ok=True)
    (output / "overlays").mkdir(exist_ok=True)
    model = GroundedSam2StopLine(device, cache, box_threshold, text_threshold)
    m2f_config = load_unified_config(Path("pnk_lane_bev/configs/segmentation/mapillary_swin_large.yaml"))
    m2f = Mask2FormerDetector(
        m2f_config["model"]["name"], device, Path("out/pnk_e2e/hf_cache")
    )
    drivable_ids = resolve_class_ids(
        MAPILLARY_ID2LABEL, m2f_config["taxonomy"]["drivable_classes"], "drivable"
    )
    lane_ids = resolve_class_ids(
        MAPILLARY_ID2LABEL, m2f_config["taxonomy"]["lane_classes"], "lane"
    )
    all_rows: list[dict] = []
    global_started = time.perf_counter()

    # Keep one SAM 2 video memory/session per camera. Text grounding runs at
    # every frame; an empty detection advances SAM 2 using its video memory.
    for camera in CAMERAS:
        session = model.sam_processor.init_video_session(
            inference_device=model.device,
            inference_state_device="cpu",
            video_storage_device="cpu",
            dtype=model.dtype,
        )
        seen_prompt = False
        for frame_idx, sample in enumerate(frames):
            camera_entry = sample["cameras"][camera]
            rgb = _rgb(camera_entry["source_path"])
            height, width = rgb.shape[:2]
            started = time.perf_counter()
            semantic_prediction = m2f.predict(rgb)
            drive_mask = np.isin(semantic_prediction.semantic, list(drivable_ids)).astype(np.uint8)
            lane_mask = np.isin(semantic_prediction.semantic, list(lane_ids)).astype(np.uint8)
            boxes, scores = model.ground(rgb)
            inp = model.sam_processor(images=rgb, return_tensors="pt").to(model.device)
            inp.pixel_values = inp.pixel_values.to(dtype=model.dtype)
            session.add_new_frame(inp.pixel_values[0], frame_idx=frame_idx)
            if boxes:
                # Stop line is a single semantic class. A union prompt box
                # lets SAM 2 refine all grounded candidates into one class mask.
                union_box = [
                    min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes),
                ]
                model.sam_processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=frame_idx,
                    obj_ids=1,
                    input_boxes=[[union_box]],
                    original_size=(height, width),
                )
                seen_prompt = True
            if seen_prompt:
                prediction = model.sam_model(inference_session=session, frame_idx=frame_idx)
                raw_masks = model.sam_processor.post_process_masks(
                    [prediction.pred_masks],
                    original_sizes=[[height, width]],
                    binarize=False,
                )[0]
                mask_prob = raw_masks[0, 0].detach().float().cpu().numpy()
                mask = _postprocess_stop(mask_prob >= 0.0)
                # If no detection was grounded at this frame, accept propagated
                # masks only while the video tracker retains visible support.
                propagated = not bool(boxes)
            else:
                mask = np.zeros((height, width), dtype=np.uint8)
                propagated = False
            stem = f"{frame_idx:05d}_{camera}"
            cv2.imwrite(str(output / "masks" / "stop_line" / f"{stem}.png"), mask * 255)
            cv2.imwrite(str(output / "masks" / "lane" / f"{stem}.png"), lane_mask * 255)
            cv2.imwrite(str(output / "masks" / "drivable" / f"{stem}.png"), drive_mask * 255)
            cv2.imwrite(
                str(output / "masks" / "semantic" / f"{stem}.png"),
                semantic_prediction.semantic.astype(np.uint16),
            )
            layered = _overlay(rgb, mask, boxes)
            drive_pixels = drive_mask.astype(bool)
            lane_pixels = lane_mask.astype(bool)
            layered[drive_pixels] = (
                0.55 * layered[drive_pixels] + 0.45 * np.array([0, 190, 0])
            ).astype(np.uint8)
            layered[lane_pixels] = (
                0.45 * layered[lane_pixels] + 0.55 * np.array([0, 255, 255])
            ).astype(np.uint8)
            cv2.imwrite(str(output / "overlays" / f"{stem}.jpg"), layered)
            all_rows.append({
                "frame_index": frame_idx,
                "timestamp_ns": int(sample["timestamp_ns"]),
                "camera": camera,
                "source_path": camera_entry["source_path"],
                "prompt": PROMPT,
                "grounding_boxes": boxes,
                "grounding_scores": scores,
                "sam2_mask_pixels": int(mask.sum()),
                "sam2_mask_fraction": float(mask.mean()),
                "mask2former_lane_pixels": int(lane_mask.sum()),
                "mask2former_drivable_pixels": int(drive_mask.sum()),
                "used_video_propagation": propagated,
                "latency_ms_grounding_plus_sam2": (time.perf_counter() - started) * 1000.0,
            })
        session.reset_inference_session()
        if model.device.type == "cuda":
            torch.cuda.empty_cache()

    detections = [row for row in all_rows if row["grounding_boxes"]]
    propagated = [row for row in all_rows if row["used_video_propagation"] and row["sam2_mask_pixels"]]
    result = {
        "schema_version": 1,
        "task": "grounded_sam2_stop_line",
        "input_raw_root": manifest["raw_root"],
        "preflight": str(preflight.resolve()),
        "frame_count": len(frames),
        "camera_count": len(CAMERAS),
        "camera_images": len(all_rows),
        "cameras": list(CAMERAS),
        "architecture": {
            "grounding_dino_model": GDINO_ID,
            "sam2_model": SAM2_ID,
            "prompt": PROMPT,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "sam2_video_memory": "one streaming session per camera; box prompt on each grounded frame",
        },
        "run": {
            "device": str(model.device),
            "wall_time_s": time.perf_counter() - global_started,
            "grounded_frame_count": len(detections),
            "grounded_frame_rate": len(detections) / max(1, len(all_rows)),
            "frames_with_nonempty_propagated_mask": len(propagated),
            "mean_mask_fraction": float(np.mean([r["sam2_mask_fraction"] for r in all_rows])),
            "mean_latency_ms": float(np.mean([r["latency_ms_grounding_plus_sam2"] for r in all_rows])),
        },
        "metrics_note": "No stop-line pixel ground truth was provided; these are coverage/runtime proxies, not accuracy metrics.",
        "per_image": all_rows,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("out/grounded_sam2/hf_cache"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--box-threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.18)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run(args.preflight, args.output, args.cache, args.device, args.limit,
                  args.box_threshold, args.text_threshold, args.force)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
