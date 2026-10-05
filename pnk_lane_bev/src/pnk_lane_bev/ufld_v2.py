"""Official UFLD-v2 CULane ResNet-18 inference adapter.

The network and decoder mirror the MIT licensed upstream implementation at the
commit recorded in constants.py.  Keeping the small adapter local avoids
depending on upstream's training-only imports at runtime.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .constants import CAMERAS, UFLD_EXISTENCE_THRESHOLD
from .io import atomic_json, load_task1_manifest, sha256_file
from .lane_types import LaneCurve, LaneDetector, LanePrediction


def _build_model(backbone_name: str = "resnet18"):
    import torch
    from torchvision.models import resnet18, resnet34

    backbones = {"resnet18": resnet18, "resnet34": resnet34}
    try:
        backbone_factory = backbones[backbone_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported UFLD-v2 backbone: {backbone_name}") from exc

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            base = backbone_factory(weights=None)
            for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"):
                setattr(self, name, getattr(base, name))

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            x2 = self.layer2(x)
            x3 = self.layer3(x2)
            return x2, x3, self.layer4(x3)

    class UFLDv2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_grid_row, self.num_cls_row = 200, 72
            self.num_grid_col, self.num_cls_col = 100, 81
            self.num_lane_on_row = self.num_lane_on_col = 4
            self.dim1 = 200 * 72 * 4
            self.dim2 = 100 * 81 * 4
            self.dim3 = 2 * 72 * 4
            self.dim4 = 2 * 81 * 4
            self.model = Backbone()
            self.pool = torch.nn.Conv2d(512, 8, 1)
            self.cls = torch.nn.Sequential(
                torch.nn.LayerNorm(4000),
                torch.nn.Linear(4000, 2048),
                torch.nn.ReLU(),
                torch.nn.Linear(2048, self.dim1 + self.dim2 + self.dim3 + self.dim4),
            )

        def forward(self, image):
            _, _, feature = self.model(image)
            out = self.cls(self.pool(feature).flatten(1))
            return {
                "loc_row": out[:, : self.dim1].view(-1, 200, 72, 4),
                "loc_col": out[:, self.dim1 : self.dim1 + self.dim2].view(-1, 100, 81, 4),
                "exist_row": out[:, self.dim1 + self.dim2 : self.dim1 + self.dim2 + self.dim3].view(-1, 2, 72, 4),
                "exist_col": out[:, -self.dim4 :].view(-1, 2, 81, 4),
            }

    return UFLDv2()


class LaneRunner:
    def __init__(self, checkpoint: Path, device: str = "auto"):
        import torch

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing UFLD-v2 checkpoint: {checkpoint}. Run download_models.ps1 first."
            )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.model = _build_model()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"UFLD checkpoint mismatch; missing={missing}, unexpected={unexpected}")
        self.model.to(self.device).eval()
        self.checkpoint = checkpoint

    @staticmethod
    def _preprocess(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1600, 533), interpolation=cv2.INTER_LINEAR)[-320:]
        tensor = image.astype(np.float32) / 255.0
        tensor = (tensor - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        return tensor.transpose(2, 0, 1)[None]

    def _decode(self, pred: dict, width: int = 768, height: int = 432) -> list[dict]:
        torch = self.torch
        row_anchor = torch.linspace(0.42, 1.0, 72)
        col_anchor = torch.linspace(0.0, 1.0, 81)
        lanes = []
        for axis, lane_indices, anchors, min_fraction in (
            ("row", [1, 2], row_anchor, 0.5),
            ("col", [0, 3], col_anchor, 0.25),
        ):
            loc = pred[f"loc_{axis}"][0].float().cpu()
            exist_prob = pred[f"exist_{axis}"][0].float().softmax(0)[1].cpu()
            valid = exist_prob >= UFLD_EXISTENCE_THRESHOLD
            grid_count = loc.shape[0]
            for lane_index in lane_indices:
                lane_valid = valid[:, lane_index]
                if int(lane_valid.sum()) <= int(valid.shape[0] * min_fraction):
                    continue
                maxima = loc[:, :, lane_index].argmax(0)
                points, confidences = [], []
                for anchor_index in torch.where(lane_valid)[0].tolist():
                    peak = int(maxima[anchor_index])
                    indexes = torch.arange(max(0, peak - 1), min(grid_count - 1, peak + 1) + 1)
                    coordinate = float(
                        (loc[indexes, anchor_index, lane_index].softmax(0) * indexes.float()).sum()
                        + 0.5
                    )
                    if axis == "row":
                        u = coordinate / (grid_count - 1) * width
                        v = float(anchors[anchor_index]) * height
                    else:
                        u = float(anchors[anchor_index]) * width
                        v = coordinate / (grid_count - 1) * height
                    if 0 <= u < width and 0 <= v < height:
                        points.append([round(u, 2), round(v, 2)])
                        confidences.append(float(exist_prob[anchor_index, lane_index]))
                if len(points) >= 4:
                    points.sort(key=lambda point: point[1])
                    lanes.append(
                        {
                            "axis": axis,
                            "slot": lane_index,
                            "points": points,
                            "confidence": float(np.mean(confidences)),
                            "point_confidence": [round(value, 6) for value in confidences],
                        }
                    )
        return lanes

    def infer(self, path: Path) -> tuple[list[dict], float]:
        torch = self.torch
        tensor = torch.from_numpy(self._preprocess(path)).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            pred = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return self._decode(pred), (time.perf_counter() - started) * 1000.0


class UFLDDetector(LaneDetector):
    """UFLD-v2 implementation of the common lane detector interface."""

    name = "ufld_v2"

    def __init__(
        self,
        checkpoint: Path,
        device: str = "auto",
        backbone: str = "resnet18",
        detector_name: str = "ufld_v2",
    ):
        import torch

        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing UFLD-v2 checkpoint: {checkpoint}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.backbone = backbone
        self.name = detector_name
        self.model = _build_model(backbone)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"UFLD checkpoint mismatch; missing={missing}, unexpected={unexpected}")
        self.model.to(self.device).eval()

    @staticmethod
    def _tensor(image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (1600, 533), interpolation=cv2.INTER_LINEAR)[-320:]
        tensor = rgb.astype(np.float32) / 255.0
        tensor = (tensor - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        return tensor.transpose(2, 0, 1)[None]

    @staticmethod
    def _curve(points: list[list[float]]) -> list[float]:
        if len(points) < 3:
            return []
        values = np.asarray(points, np.float64)
        return [float(value) for value in np.polyfit(values[:, 1], values[:, 0], min(2, len(points) - 1))]

    def _decode_prediction(self, pred: dict, width: int, height: int) -> list[LaneCurve]:
        helper = object.__new__(LaneRunner)
        helper.torch = self.torch
        records = LaneRunner._decode(helper, pred, width=width, height=height)
        return [
            LaneCurve(
                points=record["points"],
                confidence=float(record["confidence"]),
                curve=self._curve(record["points"]),
                metadata={"axis": record["axis"], "slot": record["slot"]},
            )
            for record in records
        ]

    def predict(self, image: np.ndarray) -> LanePrediction:
        torch = self.torch
        tensor = torch.from_numpy(self._tensor(image)).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            raw = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        runtime_ms = (time.perf_counter() - started) * 1000
        lanes = self._decode_prediction(raw, image.shape[1], image.shape[0])
        confidence = float(np.mean([lane.confidence for lane in lanes])) if lanes else 0.0
        return LanePrediction(
            detector=self.name,
            lanes=lanes,
            confidence=confidence,
            valid=bool(lanes),
            runtime_ms=runtime_ms,
            metadata={"input_wh": [1600, 320], "source": f"official UFLD-v2 CULane {self.backbone}"},
        )


def run_scene(
    scene_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    device: str = "auto",
    force: bool = False,
    limit: int | None = None,
) -> dict:
    manifest = load_task1_manifest(scene_dir)
    jobs = [(frame, camera) for frame in manifest["frames"] for camera in CAMERAS]
    if limit is not None:
        jobs = jobs[:limit]
    cache = output_dir / "cache" / "ufld_v2"
    pending = []
    for frame, camera in jobs:
        target = cache / camera / f"{int(frame['frame']):04d}.json"
        if force or not target.exists():
            pending.append((frame, camera, target))
    runner = LaneRunner(checkpoint, device=device) if pending else None
    checkpoint_digest = sha256_file(checkpoint)
    latency, lane_counts = [], []
    for index, (frame, camera, target) in enumerate(pending, 1):
        image = scene_dir / frame["imgs"][camera]
        lanes, elapsed_ms = runner.infer(image)
        atomic_json(
            target,
            {
                "frame": int(frame["frame"]),
                "timestamp": int(frame["timestamp"]),
                "camera": camera,
                "model": "UFLD-v2 CULane ResNet-18",
                "checkpoint_sha256": checkpoint_digest,
                "canonical_image_wh": [768, 432],
                "lanes": lanes,
                "no_lane_detected": len(lanes) == 0,
                "latency_ms": elapsed_ms,
            },
        )
        latency.append(elapsed_ms)
        lane_counts.append(len(lanes))
        print(
            f"[ufld-v2] {index}/{len(pending)} frame={frame['frame']} camera={camera} "
            f"lanes={len(lanes)} {elapsed_ms:.1f}ms",
            flush=True,
        )
    records = []
    for frame, camera in jobs:
        target = cache / camera / f"{int(frame['frame']):04d}.json"
        if target.exists():
            records.append(json.loads(target.read_text(encoding="utf-8")))
    all_latency = [float(record["latency_ms"]) for record in records]
    all_lane_counts = [len(record.get("lanes", [])) for record in records]
    summary = {
        "model": "UFLD-v2 CULane ResNet-18",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "device": str(runner.device if runner else device),
        "requested_images": len(jobs),
        "new_images": len(pending),
        "cached_images": len(jobs) - len(pending),
        "latency_ms_mean_new": float(np.mean(latency)) if latency else None,
        "mean_lanes_new": float(np.mean(lane_counts)) if lane_counts else None,
        "latency_ms_median_all": float(np.median(all_latency)) if all_latency else None,
        "latency_ms_p95_all": float(np.percentile(all_latency, 95)) if all_latency else None,
        "mean_lanes_all": float(np.mean(all_lane_counts)) if all_lane_counts else None,
        "empty_prediction_rate_all": float(np.mean(np.asarray(all_lane_counts) == 0)) if all_lane_counts else None,
    }
    atomic_json(output_dir / "ufld_summary.json", summary)
    return summary
