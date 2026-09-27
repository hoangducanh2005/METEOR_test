"""Mask2Former inference with resumable per-image panoptic caches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .taxonomy import map_mapillary_label

DEFAULT_MODEL = "facebook/mask2former-swin-large-mapillary-vistas-panoptic"


class Mask2FormerRunner:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto"):
        try:
            import torch
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        except ImportError as exc:
            raise RuntimeError(
                "Mask2Former dependencies are missing. Install the project with "
                "the [mask2former] extra."
            ) from exc
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        if self.device.type == "cuda":
            try:
                self.model = self.model.half().to(self.device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                self.model = self.model.float().to(self.device)
        else:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name

    def infer(self, image_path: Path) -> tuple[np.ndarray, list[dict]]:
        torch = self.torch
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        if self.device.type == "cuda":
            pixel_values = pixel_values.half()
        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                outputs = self.model(pixel_values=pixel_values)
        result = self.processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]
        panoptic = result["segmentation"].detach().cpu().numpy().astype(np.int32)
        segments = []
        id2label = self.model.config.id2label
        for item in result["segments_info"]:
            model_id = int(item["label_id"])
            source_label = id2label.get(model_id, id2label.get(str(model_id), str(model_id)))
            target = map_mapillary_label(str(source_label))
            if target is None:
                continue
            segments.append({
                "segment_id": int(item["id"]),
                "source_label_id": model_id,
                "source_label": str(source_label),
                "target_category": target,
                "score": float(item.get("score", 1.0)),
                "was_fused": bool(item.get("was_fused", False)),
            })
        return panoptic, segments


def run_scene(scene_dir: Path, cache_dir: Path, model_name: str = DEFAULT_MODEL,
              device: str = "auto", limit: int | None = None,
              force: bool = False) -> dict:
    manifest = json.loads((scene_dir / "run_manifest.json").read_text(encoding="utf-8"))
    jobs = []
    for sample in manifest["samples"]:
        for channel, camera in sample["cameras"].items():
            jobs.append((sample, channel, camera))
    if limit is not None:
        jobs = jobs[:limit]
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for sample, channel, camera in jobs:
        stem = f"{sample['timestamp_ns']}_{channel}"
        if (force or not (cache_dir / f"{stem}.npz").exists()
                or not (cache_dir / f"{stem}.json").exists()):
            pending.append((sample, channel, camera, stem))
    runner = Mask2FormerRunner(model_name, device) if pending else None
    mapped = 0
    for index, (sample, channel, camera, stem) in enumerate(pending, 1):
        image_path = scene_dir / camera["filename"]
        panoptic, segments = runner.infer(image_path)
        np.savez_compressed(cache_dir / f"{stem}.npz", panoptic=panoptic)
        (cache_dir / f"{stem}.json").write_text(
            json.dumps({"segments": segments, "model": model_name}, indent=2), encoding="utf-8"
        )
        mapped += len(segments)
        print(f"[mask2former] {index}/{len(pending)} {stem}: {len(segments)} mapped segments", flush=True)
    return {
        "model": model_name,
        "device": str(runner.device if runner else device),
        "requested_images": len(jobs),
        "new_images": len(pending),
        "force": force,
        "mapped_segments_in_new_images": mapped,
        "cache_dir": str(cache_dir),
    }
