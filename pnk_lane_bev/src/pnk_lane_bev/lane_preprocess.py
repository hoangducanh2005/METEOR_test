"""Raw-image undistortion and camera-specific ROI preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .constants import IMAGE_HEIGHT, IMAGE_WIDTH
from .lane_types import PreprocessedCamera


class CameraPreprocessor:
    def __init__(self, allroot_manifest: Path, camera_config: dict, debug_dir: Path | None = None):
        self.manifest_path = allroot_manifest
        self.manifest = json.loads(allroot_manifest.read_text(encoding="utf-8"))
        self.camera_config = camera_config
        self.debug_dir = debug_dir
        calibration_dir = Path(self.manifest["calibration_dir"])
        intrinsic_files = sorted(calibration_dir.glob("*_Intrinsics.json"))
        if not intrinsic_files:
            raise FileNotFoundError(f"No *_Intrinsics.json in {calibration_dir}")
        self.intrinsic_path = intrinsic_files[0]
        self.intrinsics = json.loads(self.intrinsic_path.read_text(encoding="utf-8-sig"))
        self.samples_by_timestamp = {int(sample["timestamp_ns"]): sample for sample in self.manifest["samples"]}

    def _source_record(self, timestamp: int, camera: str) -> dict:
        # Task 1 BEV manifest stores microseconds; Allroot stores nanoseconds.
        if timestamp < 100_000_000_000_000_000:
            timestamp *= 1000
        sample = self.samples_by_timestamp.get(timestamp)
        if sample is None:
            nearest = min(self.samples_by_timestamp, key=lambda value: abs(value - timestamp))
            sample = self.samples_by_timestamp[nearest]
        return sample["cameras"][camera]

    def process(self, timestamp: int, camera: str, frame_number: int, save_debug: bool = True) -> PreprocessedCamera:
        config = self.camera_config["cameras"][camera]
        source = self._source_record(timestamp, camera)
        path = Path(source["source_path"])
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError(f"Cannot decode raw camera image: {path}")
        source_channel = config["source_channel"]
        params = self.intrinsics[source_channel]
        intrinsic = np.asarray(params["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(params["distortion_coefficients"], dtype=np.float64)
        undistorted = cv2.undistort(raw, intrinsic, distortion, None, intrinsic)
        canonical = cv2.resize(undistorted, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        normalized = config["roi_normalized"]
        x0 = int(round(normalized[0] * IMAGE_WIDTH))
        y0 = int(round(normalized[1] * IMAGE_HEIGHT))
        x1 = int(round(normalized[2] * IMAGE_WIDTH))
        y1 = int(round(normalized[3] * IMAGE_HEIGHT))
        roi = canonical[y0:y1, x0:x1].copy()
        result = PreprocessedCamera(
            camera=camera,
            raw_image=raw,
            undistorted_image=undistorted,
            canonical_image=canonical,
            roi_image=roi,
            roi_xyxy=(x0, y0, x1, y1),
            source_path=str(path),
            intrinsic_source=str(self.intrinsic_path),
            distortion_source=source_channel,
        )
        if self.debug_dir is not None and save_debug:
            self.save_debug(result, frame_number)
        return result

    def save_debug(self, value: PreprocessedCamera, frame_number: int) -> None:
        target = self.debug_dir / value.camera
        target.mkdir(parents=True, exist_ok=True)
        raw_small = cv2.resize(value.raw_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        undistorted_small = cv2.resize(
            value.undistorted_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA
        )
        roi_marked = value.canonical_image.copy()
        x0, y0, x1, y1 = value.roi_xyxy
        cv2.rectangle(roi_marked, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 3)
        panel = np.hstack((raw_small, undistorted_small, roi_marked))
        labels = ("RAW", "UNDISTORTED", "CAMERA ROI")
        for index, label in enumerate(labels):
            cv2.putText(
                panel,
                label,
                (index * IMAGE_WIDTH + 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(target / f"{frame_number:04d}_preprocess.jpg"), panel)
