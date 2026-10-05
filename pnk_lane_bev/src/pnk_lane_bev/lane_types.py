"""Shared contracts for interchangeable lane detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class LaneCurve:
    points: list[list[float]]
    confidence: float
    curve: list[float] = field(default_factory=list)
    valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LanePrediction:
    detector: str
    lanes: list[LaneCurve]
    confidence: float
    valid: bool
    runtime_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def lane_points(self) -> list[list[list[float]]]:
        return [lane.points for lane in self.lanes]

    @property
    def lane_curves(self) -> list[list[float]]:
        return [lane.curve for lane in self.lanes]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["lane_points"] = self.lane_points
        value["lane_curves"] = self.lane_curves
        return value


@dataclass
class PreprocessedCamera:
    camera: str
    raw_image: np.ndarray
    undistorted_image: np.ndarray
    canonical_image: np.ndarray
    roi_image: np.ndarray
    roi_xyxy: tuple[int, int, int, int]
    source_path: str
    intrinsic_source: str
    distortion_source: str

    def roi_to_canonical(self, points: list[list[float]]) -> list[list[float]]:
        x0, y0, x1, y1 = self.roi_xyxy
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        roi_h, roi_w = self.roi_image.shape[:2]
        return [
            [float(x0 + point[0] * width / roi_w), float(y0 + point[1] * height / roi_h)]
            for point in points
        ]


class LaneDetector(ABC):
    """Detector interface used by inference, benchmark and fine-tune adapters."""

    name: str

    @abstractmethod
    def predict(self, image: np.ndarray) -> LanePrediction:
        """Predict lanes from a BGR image and return points in that image space."""

