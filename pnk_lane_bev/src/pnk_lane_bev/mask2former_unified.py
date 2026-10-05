"""One Mask2Former model for drivable area and lane-marking segmentation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .constants import CAMERAS, IMAGE_HEIGHT, IMAGE_WIDTH
from .io import atomic_json, atomic_npz, load_task1_manifest, read_json
from .unified_config import load_unified_config, resolve_class_ids


MAPILLARY_ID2LABEL = {
    0: "Bird", 1: "Ground Animal", 2: "Curb", 3: "Fence", 4: "Guard Rail",
    5: "Barrier", 6: "Wall", 7: "Bike Lane", 8: "Crosswalk - Plain",
    9: "Curb Cut", 10: "Parking", 11: "Pedestrian Area", 12: "Rail Track",
    13: "Road", 14: "Service Lane", 15: "Sidewalk", 16: "Bridge",
    17: "Building", 18: "Tunnel", 19: "Person", 20: "Bicyclist",
    21: "Motorcyclist", 22: "Other Rider", 23: "Lane Marking - Crosswalk",
    24: "Lane Marking - General", 25: "Mountain", 26: "Sand", 27: "Sky",
    28: "Snow", 29: "Terrain", 30: "Vegetation", 31: "Water", 32: "Banner",
    33: "Bench", 34: "Bike Rack", 35: "Billboard", 36: "Catch Basin",
    37: "CCTV Camera", 38: "Fire Hydrant", 39: "Junction Box", 40: "Mailbox",
    41: "Manhole", 42: "Phone Booth", 43: "Pothole", 44: "Street Light",
    45: "Pole", 46: "Traffic Sign Frame", 47: "Utility Pole", 48: "Traffic Light",
    49: "Traffic Sign (Back)", 50: "Traffic Sign (Front)", 51: "Trash Can",
    52: "Bicycle", 53: "Boat", 54: "Bus", 55: "Car", 56: "Caravan",
    57: "Motorcycle", 58: "On Rails", 59: "Other Vehicle", 60: "Trailer",
    61: "Truck", 62: "Wheeled Slow", 63: "Car Mount", 64: "Ego Vehicle",
}


@dataclass
class SegmentationPrediction:
    panoptic: np.ndarray
    semantic: np.ndarray
    semantic_confidence: np.ndarray
    segments: list[dict]
    latency_ms: float


def semantic_from_panoptic(
    panoptic: np.ndarray, segments: list[dict], id2label: dict[int, str]
) -> tuple[np.ndarray, np.ndarray]:
    semantic = np.full(panoptic.shape, 65535, dtype=np.uint16)
    confidence = np.zeros(panoptic.shape, dtype=np.float32)
    for segment in segments:
        segment_id = int(segment.get("id", segment.get("segment_id")))
        label_id = int(segment.get("label_id", segment.get("source_label_id")))
        if label_id not in id2label:
            continue
        pixels = panoptic == segment_id
        semantic[pixels] = label_id
        confidence[pixels] = float(segment.get("score", 1.0))
    return semantic, confidence


class Mask2FormerDetector:
    """RGB -> panoptic/semantic map using the configured Mapillary checkpoint."""

    def __init__(self, model_name: str, device: str = "auto", cache_dir: Path | None = None):
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
        source: str | Path = model_name
        if cache_dir is not None:
            model_cache = (
                cache_dir / "hub" /
                f"models--{model_name.replace('/', '--')}" / "snapshots"
            )
            snapshots = sorted(path for path in model_cache.glob("*") if path.is_dir())
            if snapshots:
                source = snapshots[-1]
                kwargs = {}
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                source, local_files_only=True, **kwargs
            )
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                source, local_files_only=True, **kwargs
            )
        except OSError:
            if isinstance(source, Path):
                raise
            self.processor = AutoImageProcessor.from_pretrained(model_name, **kwargs)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, **kwargs)
        if self.device.type == "cuda":
            self.model = self.model.half().to(self.device)
        else:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name
        self.id2label = {int(key): str(value) for key, value in self.model.config.id2label.items()}
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def predict(self, image_rgb: np.ndarray) -> SegmentationPrediction:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB HxWx3, got {image_rgb.shape}")
        torch = self.torch
        image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        if self.device.type == "cuda":
            pixel_values = pixel_values.half()
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        ):
            outputs = self.model(pixel_values=pixel_values)
        result = self.processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        panoptic = result["segmentation"].detach().cpu().numpy().astype(np.int32)
        segments = []
        for item in result["segments_info"]:
            label_id = int(item["label_id"])
            segments.append(
                {
                    "segment_id": int(item["id"]),
                    "source_label_id": label_id,
                    "source_label": self.id2label.get(label_id, str(label_id)),
                    "score": float(item.get("score", 1.0)),
                    "was_fused": bool(item.get("was_fused", False)),
                }
            )
        semantic, confidence = semantic_from_panoptic(panoptic, segments, self.id2label)
        return SegmentationPrediction(panoptic, semantic, confidence, segments, latency_ms)


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    image = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image):
        eroded = cv2.erode(image, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = eroded
    return (skeleton > 0).astype(np.uint8)


def postprocess_lane(mask: np.ndarray, options: dict) -> np.ndarray:
    result = (mask > 0).astype(np.uint8)
    close_kernel = int(options.get("close_kernel", 1))
    open_kernel = int(options.get("open_kernel", 1))
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    if open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    minimum = int(options.get("min_component_pixels", 0))
    if minimum > 1 and result.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(result, 8)
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum
        result = keep[labels].astype(np.uint8)
    if bool(options.get("skeletonize", False)):
        result = _skeletonize(result)
    return result


def _build_masks(
    prediction: SegmentationPrediction,
    drivable_ids: set[int],
    lane_ids: set[int],
    lane_threshold: float,
    lane_options: dict,
) -> dict[str, np.ndarray]:
    semantic = prediction.semantic
    confidence = prediction.semantic_confidence
    drivable_mask = np.isin(semantic, list(drivable_ids)).astype(np.uint8)
    lane_class = np.isin(semantic, list(lane_ids))
    lane_mask_raw = (lane_class & (confidence >= lane_threshold)).astype(np.uint8)
    lane_mask_post = postprocess_lane(lane_mask_raw, lane_options)
    return {
        "panoptic": prediction.panoptic.astype(np.int32),
        "semantic": semantic.astype(np.uint16),
        "semantic_confidence": confidence.astype(np.float16),
        "drivable_mask": drivable_mask,
        "drivable_probability": (confidence * drivable_mask).astype(np.float16),
        "lane_mask_raw": lane_mask_raw,
        "lane_mask_post": lane_mask_post,
        "lane_probability_raw": (confidence * lane_mask_raw).astype(np.float16),
        "lane_probability_post": (confidence * lane_mask_post).astype(np.float16),
    }


def _nearest_sample(allroot: dict, timestamp_us: int) -> dict:
    target = int(timestamp_us) * 1000
    return min(allroot["samples"], key=lambda sample: abs(int(sample["timestamp_ns"]) - target))


def run_inference(
    scene_dir: Path,
    output_dir: Path,
    config_path: Path,
    allroot_manifest: Path,
    source_panoptic_cache: Path | None = None,
    device: str = "auto",
    force: bool = False,
    limit: int | None = None,
) -> dict:
    config = load_unified_config(config_path)
    manifest = load_task1_manifest(scene_dir)
    frames = manifest["frames"][:limit] if limit is not None else manifest["frames"]
    allroot = read_json(allroot_manifest)
    requested_labels = config["taxonomy"]
    id2label = MAPILLARY_ID2LABEL
    drivable = resolve_class_ids(id2label, requested_labels["drivable_classes"], "drivable_classes")
    lane = resolve_class_ids(id2label, requested_labels["lane_classes"], "lane_classes")
    cache_root = output_dir / "cache" / "mask2former_unified"
    jobs = []
    for frame in frames:
        sample = _nearest_sample(allroot, int(frame["timestamp"]))
        for camera in CAMERAS:
            target_npz = cache_root / camera / f"{int(frame['frame']):04d}.npz"
            target_json = cache_root / camera / f"{int(frame['frame']):04d}.json"
            if force or not target_npz.exists() or not target_json.exists():
                jobs.append((frame, sample, camera, target_npz, target_json))
    detector = None
    new_direct = 0
    new_reused = 0
    post_latency = []
    inference_latency = []
    for index, (frame, sample, camera, target_npz, target_json) in enumerate(jobs, 1):
        source_stem = f"{int(sample['timestamp_ns'])}_{camera}"
        source_npz = source_panoptic_cache / f"{source_stem}.npz" if source_panoptic_cache else None
        source_json = source_panoptic_cache / f"{source_stem}.json" if source_panoptic_cache else None
        if source_npz is not None and source_npz.exists() and source_json.exists():
            with np.load(source_npz) as data:
                panoptic = data["panoptic"].astype(np.int32)
            source_record = read_json(source_json)
            segments = source_record["segments"]
            semantic, confidence = semantic_from_panoptic(panoptic, segments, id2label)
            prediction = SegmentationPrediction(panoptic, semantic, confidence, segments, 0.0)
            backend = "reused_same_checkpoint_panoptic_cache"
            new_reused += 1
        else:
            if detector is None:
                detector = Mask2FormerDetector(
                    config["model"]["name"], device=device,
                    cache_dir=Path(config["model"]["local_cache_dir"]),
                )
                id2label = detector.id2label
                drivable = resolve_class_ids(id2label, requested_labels["drivable_classes"], "drivable_classes")
                lane = resolve_class_ids(id2label, requested_labels["lane_classes"], "lane_classes")
            bgr = cv2.imread(str(scene_dir / frame["imgs"][camera]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(scene_dir / frame["imgs"][camera])
            prediction = detector.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            backend = "direct_model_inference"
            new_direct += 1
            inference_latency.append(prediction.latency_ms)
        started = time.perf_counter()
        threshold = float(config["cameras"][camera]["lane_confidence_threshold"])
        arrays = _build_masks(
            prediction, set(drivable), set(lane), threshold, config["postprocess"]["lane"]
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        post_latency.append(elapsed)
        atomic_npz(target_npz, **arrays)
        atomic_json(
            target_json,
            {
                "frame": int(frame["frame"]),
                "timestamp": int(frame["timestamp"]),
                "camera": camera,
                "model": config["model"]["name"],
                "model_revision": getattr(detector, "revision", None),
                "inference_backend": backend,
                "source_panoptic_cache": str(source_npz.resolve()) if backend.startswith("reused") else None,
                "inference_latency_ms": prediction.latency_ms if backend == "direct_model_inference" else None,
                "postprocess_latency_ms": elapsed,
                "drivable_class_ids": drivable,
                "lane_class_ids": lane,
                "lane_confidence_threshold": threshold,
                "drivable_fraction": float(arrays["drivable_mask"].mean()),
                "lane_fraction_raw": float(arrays["lane_mask_raw"].mean()),
                "lane_fraction_post": float(arrays["lane_mask_post"].mean()),
                "segments": prediction.segments,
            },
        )
        print(
            f"[mask2former-unified] {index}/{len(jobs)} frame={frame['frame']} "
            f"camera={camera} backend={backend} lane={arrays['lane_mask_post'].mean():.4f}",
            flush=True,
        )
    records = [
        read_json(cache_root / camera / f"{int(frame['frame']):04d}.json")
        for frame in frames for camera in CAMERAS
    ]
    summary = {
        "model": config["model"]["name"],
        "inference_path": "RGB -> Mask2Former -> drivable+lane (no UFLD-v2)",
        "requested_images": len(frames) * len(CAMERAS),
        "new_images": len(jobs),
        "new_direct_inference": new_direct,
        "new_reused_same_checkpoint_cache": new_reused,
        "cached_images": len(frames) * len(CAMERAS) - len(jobs),
        "drivable_classes": list(drivable.values()),
        "lane_classes": list(lane.values()),
        "postprocess_latency_ms_mean_new": float(np.mean(post_latency)) if post_latency else None,
        "direct_inference_latency_ms_mean_new": float(np.mean(inference_latency)) if inference_latency else None,
        "empty_lane_rate": float(np.mean([record["lane_fraction_post"] == 0 for record in records])),
    }
    atomic_json(output_dir / "inference_summary.json", summary)
    return summary


def benchmark_inference(
    scene_dir: Path,
    output_dir: Path,
    config_path: Path,
    device: str = "auto",
    frames_per_camera: int = 2,
) -> dict:
    config = load_unified_config(config_path)
    manifest = load_task1_manifest(scene_dir)
    detector = Mask2FormerDetector(
        config["model"]["name"], device=device,
        cache_dir=Path(config["model"]["local_cache_dir"]),
    )
    rows = []
    # Warm up kernels and allocator without including the sample in the report.
    first = manifest["frames"][0]
    warmup = cv2.imread(str(scene_dir / first["imgs"][CAMERAS[0]]), cv2.IMREAD_COLOR)
    detector.predict(cv2.cvtColor(warmup, cv2.COLOR_BGR2RGB))
    for camera in CAMERAS:
        for frame in manifest["frames"][:frames_per_camera]:
            image = cv2.imread(str(scene_dir / frame["imgs"][camera]), cv2.IMREAD_COLOR)
            prediction = detector.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            rows.append({"camera": camera, "frame": int(frame["frame"]), "latency_ms": prediction.latency_ms})
            print(f"[mask2former-benchmark] {camera} frame={frame['frame']} {prediction.latency_ms:.1f}ms", flush=True)
    by_camera = {}
    for camera in CAMERAS:
        values = np.asarray([row["latency_ms"] for row in rows if row["camera"] == camera])
        by_camera[camera] = {
            "samples": int(len(values)), "mean_ms": float(values.mean()),
            "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
        }
    values = np.asarray([row["latency_ms"] for row in rows])
    summary = {
        "model": config["model"]["name"], "device": str(detector.device),
        "warmup_samples": 1, "timed_samples": len(rows),
        "mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)), "per_camera": by_camera, "rows": rows,
    }
    atomic_json(output_dir / "inference_benchmark.json", summary)
    return summary
