"""Mask2Former Swin-Small semantic road inference with resumable caches."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import CAMERAS, MASK2FORMER_DRIVABLE_THRESHOLD, MASK2FORMER_MODEL
from .io import atomic_json, atomic_npz, load_task1_manifest, read_json


def _cached_revision(output_dir: Path) -> str | None:
    snapshots = (
        output_dir
        / "hf_cache"
        / "hub"
        / "models--facebook--mask2former-swin-small-cityscapes-semantic"
        / "snapshots"
    )
    revisions = sorted(path.name for path in snapshots.glob("*") if path.is_dir())
    return revisions[-1] if revisions else None


class DrivableRunner:
    def __init__(self, device: str = "auto", model_name: str = MASK2FORMER_MODEL):
        import torch
        import torch.nn.functional as functional
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.torch = torch
        self.functional = functional
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            # First run downloads the pinned public checkpoint; later runs stay offline.
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        if self.device.type == "cuda":
            self.model = self.model.half().to(self.device)
        else:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name
        self.revision = getattr(self.model.config, "_commit_hash", None)
        labels = {int(key): str(value) for key, value in self.model.config.id2label.items()}
        road_ids = [idx for idx, label in labels.items() if label.strip().lower() == "road"]
        if len(road_ids) != 1:
            raise ValueError(f"Expected exactly one Cityscapes road class, got {road_ids} from {labels}")
        self.road_id = road_ids[0]

    def infer(self, image_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
        torch = self.torch
        image = Image.open(image_path).convert("RGB")
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
            class_prob = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
            mask_prob = outputs.masks_queries_logits.sigmoid()
            semantic_score = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)
            semantic_score = self.functional.interpolate(
                semantic_score,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )
            normalized = semantic_score / semantic_score.sum(dim=1, keepdim=True).clamp_min(1e-6)
            semantic = semantic_score.argmax(dim=1)
            road_probability = normalized[:, self.road_id]
            road_mask = (semantic == self.road_id) | (
                road_probability >= MASK2FORMER_DRIVABLE_THRESHOLD
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return (
            road_mask[0].detach().cpu().numpy().astype(np.uint8),
            road_probability[0].detach().float().cpu().numpy().astype(np.float32),
            elapsed_ms,
        )


def run_scene(
    scene_dir: Path,
    output_dir: Path,
    device: str = "auto",
    force: bool = False,
    limit: int | None = None,
) -> dict:
    manifest = load_task1_manifest(scene_dir)
    jobs = []
    for frame in manifest["frames"]:
        for camera in CAMERAS:
            jobs.append((frame, camera))
    if limit is not None:
        jobs = jobs[:limit]
    cache_root = output_dir / "cache" / "mask2former_small"
    pending = []
    for frame, camera in jobs:
        stem = f"{int(frame['frame']):04d}"
        npz = cache_root / camera / f"{stem}.npz"
        meta = cache_root / camera / f"{stem}.json"
        if force or not npz.exists() or not meta.exists():
            pending.append((frame, camera, npz, meta))

    runner = DrivableRunner(device=device) if pending else None
    latency = []
    for index, (frame, camera, npz, meta) in enumerate(pending, 1):
        path = scene_dir / frame["imgs"][camera]
        mask, probability, elapsed_ms = runner.infer(path)
        atomic_npz(npz, mask=mask, probability=probability.astype(np.float16))
        atomic_json(
            meta,
            {
                "frame": int(frame["frame"]),
                "timestamp": int(frame["timestamp"]),
                "camera": camera,
                "model": MASK2FORMER_MODEL,
                "model_revision": runner.revision,
                "road_label_id": runner.road_id,
                "threshold": MASK2FORMER_DRIVABLE_THRESHOLD,
                "latency_ms": elapsed_ms,
                "drivable_fraction": float(mask.mean()),
                "reference_type": "prediction",
            },
        )
        latency.append(elapsed_ms)
        print(
            f"[mask2former-small] {index}/{len(pending)} frame={frame['frame']} "
            f"camera={camera} road={mask.mean():.3f} {elapsed_ms:.1f}ms",
            flush=True,
        )
    records = []
    for frame, camera in jobs:
        meta = cache_root / camera / f"{int(frame['frame']):04d}.json"
        if meta.exists():
            records.append(read_json(meta))
    all_latency = [float(record["latency_ms"]) for record in records]
    previous_summary_path = output_dir / "mask2former_summary.json"
    previous_summary = read_json(previous_summary_path) if previous_summary_path.exists() else {}
    summary = {
        "model": MASK2FORMER_MODEL,
        "model_revision": runner.revision if runner else _cached_revision(output_dir),
        "device": str(runner.device if runner else previous_summary.get("device", device)),
        "requested_images": len(jobs),
        "new_images": len(pending),
        "cached_images": len(jobs) - len(pending),
        "latency_ms_mean_new": float(np.mean(latency)) if latency else None,
        "latency_ms_median_all": float(np.median(all_latency)) if all_latency else None,
        "latency_ms_p95_all": float(np.percentile(all_latency, 95)) if all_latency else None,
        "drivable_fraction_mean_all": float(np.mean([record["drivable_fraction"] for record in records])) if records else None,
    }
    atomic_json(output_dir / "mask2former_summary.json", summary)
    return summary
