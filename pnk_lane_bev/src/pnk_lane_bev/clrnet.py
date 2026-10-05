"""CLRNet ResNet-18 inference adapter without the legacy MMCV/CUDA extension.

The module mirrors the official Apache-2.0 model graph.  Lane NMS is expressed
in plain PyTorch so the released checkpoint can run on current Windows/Torch.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from .lane_types import LaneCurve, LaneDetector, LanePrediction


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, *, padding=0, bias=True, norm=False, activate=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        if norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.activate = activate

    def forward(self, value):
        value = self.conv(value)
        if hasattr(self, "bn"):
            value = self.bn(value)
        return F.relu(value, inplace=True) if self.activate else value


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()

    def forward(self, value):
        model = self.model
        value = model.maxpool(model.relu(model.bn1(model.conv1(value))))
        outputs = []
        for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
            value = layer(value)
            outputs.append(value)
        return outputs


class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [ConvModule(channels, 64, 1) for channels in (128, 256, 512)]
        )
        self.fpn_convs = nn.ModuleList([ConvModule(64, 64, 3, padding=1) for _ in range(3)])

    def forward(self, inputs):
        inputs = inputs[-3:]
        laterals = [module(value) for module, value in zip(self.lateral_convs, inputs)]
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] += F.interpolate(laterals[index], size=laterals[index - 1].shape[2:], mode="nearest")
        return tuple(module(value) for module, value in zip(self.fpn_convs, laterals))


class SegDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(192, 5, 1)

    def forward(self, value):
        return F.interpolate(self.conv(value), size=(320, 800), mode="bilinear", align_corners=False)


class FeatureResize(nn.Module):
    def forward(self, value):
        return F.interpolate(value, (10, 25)).flatten(2)


class ROIGather(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        self.num_priors = 192
        self.f_key = ConvModule(64, 64, 1, bias=False, norm=True, activate=True)
        self.f_query = nn.Sequential(nn.Conv1d(192, 192, 1, groups=192), nn.ReLU())
        self.f_value = nn.Conv2d(64, 64, 1)
        self.W = nn.Conv1d(192, 192, 1, groups=192)
        self.resize = FeatureResize()
        self.convs = nn.ModuleList(
            [ConvModule(64, 48, (9, 1), padding=(4, 0), bias=False, norm=True, activate=True) for _ in range(3)]
        )
        self.catconv = nn.ModuleList(
            [
                ConvModule(48 * (index + 1), 64, (9, 1), padding=(4, 0), bias=False, norm=True, activate=True)
                for index in range(3)
            ]
        )
        self.fc = nn.Linear(36 * 64, 64)
        self.fc_norm = nn.LayerNorm(64)

    def forward(self, roi_features, feature, layer_index):
        values = [self.convs[index](value) for index, value in enumerate(roi_features)]
        roi = self.catconv[layer_index](torch.cat(values, dim=1))
        batch_size = feature.size(0)
        roi = roi.contiguous().view(batch_size * self.num_priors, -1)
        roi = F.relu(self.fc_norm(self.fc(roi))).view(batch_size, self.num_priors, -1)
        query = self.f_query(roi)
        key = self.resize(self.f_key(feature))
        value = self.resize(self.f_value(feature)).permute(0, 2, 1)
        similarity = F.softmax((self.in_channels**-0.5) * torch.matmul(query, key), dim=-1)
        context = self.W(torch.matmul(similarity, value))
        return roi + F.dropout(context, p=0.1, training=self.training)


def _linear_modules():
    return nn.ModuleList([nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True)])


class CLRHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_w, self.img_h = 800, 320
        self.n_offsets, self.n_strips = 72, 71
        self.num_priors, self.sample_points, self.refine_layers = 192, 36, 3
        sample_indexes = (torch.linspace(0, 1, steps=36) * self.n_strips).long()
        self.register_buffer("sample_x_indexs", sample_indexes, persistent=False)
        self.register_buffer("prior_feat_ys", torch.flip(1 - sample_indexes.float() / self.n_strips, dims=[-1]), persistent=False)
        self.register_buffer("prior_ys", torch.linspace(1, 0, steps=72), persistent=False)
        self.prior_embeddings = nn.Embedding(192, 3)
        priors, feature_priors = self._initial_priors()
        self.register_buffer("priors", priors)
        self.register_buffer("priors_on_featmap", feature_priors)
        self.seg_decoder = SegDecoder()
        self.reg_modules = _linear_modules()
        self.cls_modules = _linear_modules()
        self.roi_gather = ROIGather()
        self.reg_layers = nn.Linear(64, 76)
        self.cls_layers = nn.Linear(64, 2)

    def _initial_priors(self):
        weights = self.prior_embeddings.weight
        priors = weights.new_zeros((192, 78))
        priors[:, 2:5] = weights.clone()
        ys = self.prior_ys.repeat(192, 1)
        priors[:, 6:] = (
            priors[:, 3, None] * 799
            + ((1 - ys - priors[:, 2, None]) * 320 / torch.tan(priors[:, 4, None] * math.pi + 1e-5))
        ) / 799
        return priors, priors[..., 6 + self.sample_x_indexs]

    def _pool(self, features, prior_xs):
        batch_size = features.shape[0]
        xs = prior_xs.view(batch_size, 192, -1, 1) * 2 - 1
        ys = self.prior_feat_ys.repeat(batch_size * 192).view(batch_size, 192, -1, 1) * 2 - 1
        grid = torch.cat((xs, ys), dim=-1)
        sampled = F.grid_sample(features, grid, align_corners=True).permute(0, 2, 1, 3)
        return sampled.reshape(batch_size * 192, 64, 36, 1)

    def forward(self, features):
        features = list(features[-3:])
        features.reverse()
        batch_size = features[-1].shape[0]
        priors = self.priors.repeat(batch_size, 1, 1)
        priors_feature = self.priors_on_featmap.repeat(batch_size, 1, 1)
        stage_features = []
        output = None
        for stage in range(3):
            pooled = self._pool(features[stage], torch.flip(priors_feature, dims=[2]))
            stage_features.append(pooled)
            values = self.roi_gather(stage_features, features[stage], stage)
            values = values.view(192, batch_size, -1).reshape(batch_size * 192, 64)
            cls_value, reg_value = values.clone(), values.clone()
            for layer in self.cls_modules:
                cls_value = layer(cls_value)
            for layer in self.reg_modules:
                reg_value = layer(reg_value)
            cls_logits = self.cls_layers(cls_value).reshape(batch_size, 192, 2)
            regression = self.reg_layers(reg_value).reshape(batch_size, 192, 76)
            predictions = priors.clone()
            predictions[..., :2] = cls_logits
            predictions[..., 2:5] += regression[..., :3]
            predictions[..., 5] = regression[..., 3]
            repeat = lambda value: value.unsqueeze(2).repeat(1, 1, 72)
            predictions[..., 6:] = (
                repeat(predictions[..., 3]) * 799
                + (
                    (1 - self.prior_ys.repeat(batch_size, 192, 1) - repeat(predictions[..., 2]))
                    * 320
                    / torch.tan(repeat(predictions[..., 4]) * math.pi + 1e-5)
                )
            ) / 799
            prediction_lines = predictions.clone()
            predictions[..., 6:] += regression[..., 4:]
            output = predictions
            if stage != 2:
                priors = prediction_lines.detach().clone()
                priors_feature = priors[..., 6 + self.sample_x_indexs]
        return output


class CLRNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.neck = FPN()
        self.heads = CLRHead()

    def forward(self, image):
        return self.heads(self.neck(self.backbone(image)))


def _lane_nms(predictions: torch.Tensor, scores: torch.Tensor, threshold: float, top_k: int) -> list[int]:
    """Equivalent to CLRNet's line-distance NMS, applied to at most 192 priors."""
    order = scores.argsort(descending=True).tolist()
    kept: list[int] = []
    for candidate in order:
        lane = predictions[candidate]
        suppress = False
        for selected in kept:
            other = predictions[selected]
            start = max(round(float(lane[2]) * 71), round(float(other[2]) * 71))
            end_lane = round(float(lane[2]) * 71) + round(float(lane[5]) * 71) - 1
            end_other = round(float(other[2]) * 71) + round(float(other[5]) * 71) - 1
            end = min(end_lane, end_other, 71)
            if end >= start:
                distance = torch.abs(lane[6 + start : 7 + end] - other[6 + start : 7 + end]).sum() * 799
                if float(distance) < threshold * (end - start + 1):
                    suppress = True
                    break
        if not suppress:
            kept.append(candidate)
            if len(kept) == top_k:
                break
    return kept


class CLRNetDetector(LaneDetector):
    name = "clrnet"

    def __init__(self, checkpoint: Path, device: str = "auto", proposal_threshold: float = 0.4):
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing CLRNet checkpoint: {checkpoint}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = CLRNetModel()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = {key.removeprefix("module."): value for key, value in payload["net"].items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        allowed_missing = {"heads.sample_x_indexs", "heads.prior_feat_ys", "heads.prior_ys"}
        if set(missing) - allowed_missing or unexpected:
            raise RuntimeError(f"CLRNet checkpoint mismatch; missing={missing}, unexpected={unexpected}")
        self.model.to(self.device).eval()
        self.proposal_threshold = proposal_threshold

    @staticmethod
    def _preprocess(image: np.ndarray) -> torch.Tensor:
        height = image.shape[0]
        cut = int(round(height * 270 / 590))
        cropped = image[cut:, :, :]
        resized = cv2.resize(cropped, (800, 320), interpolation=cv2.INTER_LINEAR)
        # Official GenerateLaneLine retains OpenCV BGR order and scales to [0, 1].
        return torch.from_numpy((resized.astype(np.float32) / 255.0).transpose(2, 0, 1).copy())[None]

    @staticmethod
    def _curve(points: list[list[float]]) -> list[float]:
        if len(points) < 3:
            return []
        array = np.asarray(points, np.float64)
        coefficients = np.polyfit(array[:, 1], array[:, 0], min(2, len(points) - 1))
        return [float(value) for value in coefficients]

    def _decode(self, output: torch.Tensor, width: int, height: int) -> list[LaneCurve]:
        predictions = output[0]
        scores = predictions[:, :2].softmax(dim=1)[:, 1]
        mask = scores >= self.proposal_threshold
        predictions, scores = predictions[mask], scores[mask]
        if not len(predictions):
            return []
        keep = _lane_nms(predictions, scores, threshold=50.0, top_k=4)
        lanes = []
        prior_ys = torch.linspace(1, 0, 72, device=predictions.device)
        for index in keep:
            lane = predictions[index].clone()
            start = min(max(0, round(float(lane[2]) * 71)), 71)
            length = round(float(lane[5]) * 71)
            end = min(start + length - 1, 71)
            xs = lane[6:].clone()
            if start > 0:
                in_bounds = ((xs[:start] >= 0) & (xs[:start] <= 1)).detach().cpu().numpy()
                keep_extension = np.cumprod(in_bounds[::-1])[::-1].astype(bool)
                xs[:start][torch.from_numpy(~keep_extension).to(xs.device)] = -2
            xs[end + 1 :] = -2
            valid = (xs >= 0) & (xs <= 1)
            x_values = xs[valid].flip(0)
            y_values = prior_ys[valid].flip(0)
            if len(x_values) <= 1:
                continue
            # Official CULane output maps the 270px crop back into the original image.
            y_values = (y_values * (590 - 270) + 270) / 590
            points = [
                [float(x * width), float(y * height)]
                for x, y in zip(x_values.detach().cpu(), y_values.detach().cpu())
                if 0 <= float(x) <= 1 and 0 <= float(y) <= 1
            ]
            confidence = float(scores[index])
            if len(points) >= 2:
                lanes.append(
                    LaneCurve(
                        points=points,
                        confidence=confidence,
                        curve=self._curve(points),
                        metadata={"proposal_threshold": self.proposal_threshold},
                    )
                )
        return lanes

    def predict(self, image: np.ndarray) -> LanePrediction:
        tensor = self._preprocess(image).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        runtime_ms = (time.perf_counter() - started) * 1000
        lanes = self._decode(output, image.shape[1], image.shape[0])
        confidence = float(np.mean([lane.confidence for lane in lanes])) if lanes else 0.0
        return LanePrediction(
            detector=self.name,
            lanes=lanes,
            confidence=confidence,
            valid=bool(lanes),
            runtime_ms=runtime_ms,
            metadata={"input_wh": [800, 320], "source": "official CLRNet CULane ResNet-18"},
        )
