"""Calibrated camera to ground-plane geometry for the Task 2 BEV."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .constants import (
    BEV_FORWARD,
    BEV_HALF_WIDTH,
    BEV_HEIGHT,
    BEV_RESOLUTION,
    BEV_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)


@dataclass
class CameraGeometry:
    name: str
    map_x: np.ndarray
    map_y: np.ndarray
    valid: np.ndarray
    weight: np.ndarray
    ground_to_image: np.ndarray
    image_to_ground: np.ndarray
    metrics: dict


def bev_ground_grid() -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices((BEV_HEIGHT, BEV_WIDTH), dtype=np.float64)
    x = BEV_FORWARD - (rows + 0.5) * BEV_RESOLUTION
    y = BEV_HALF_WIDTH - (cols + 0.5) * BEV_RESOLUTION
    return x, y


def build_camera_geometry(name: str, calibration: dict) -> CameraGeometry:
    intrinsic = np.asarray(calibration["K"], dtype=np.float64)
    # Task 1 field name is retained for compatibility. Its values encode camera -> ego.
    camera_to_ego = np.asarray(calibration["T_ego_cam"], dtype=np.float64)
    ego_to_camera = np.linalg.inv(camera_to_ego)
    rotation, translation = ego_to_camera[:3, :3], ego_to_camera[:3, 3]
    ground_to_image = intrinsic @ np.column_stack((rotation[:, 0], rotation[:, 1], translation))
    image_to_ground = np.linalg.inv(ground_to_image)

    x, y = bev_ground_grid()
    ground = np.stack((x, y, np.ones_like(x)), axis=-1)
    projected = ground @ ground_to_image.T
    depth = projected[..., 2]
    map_x = projected[..., 0] / np.where(np.abs(depth) > 1e-9, depth, 1.0)
    map_y = projected[..., 1] / np.where(np.abs(depth) > 1e-9, depth, 1.0)
    valid = (
        (depth > 0.5)
        & (map_x >= 1)
        & (map_x < IMAGE_WIDTH - 1)
        & (map_y >= 1)
        & (map_y < IMAGE_HEIGHT - 1)
    )
    edge = np.minimum.reduce(
        [map_x / IMAGE_WIDTH, (IMAGE_WIDTH - map_x) / IMAGE_WIDTH, map_y / IMAGE_HEIGHT, (IMAGE_HEIGHT - map_y) / IMAGE_HEIGHT]
    )
    edge = np.clip(edge * 5.0, 0.0, 1.0)
    distance = np.sqrt(x * x + y * y)
    distance_weight = np.clip(1.0 - distance / 100.0, 0.15, 1.0)
    weight = (valid * edge * distance_weight).astype(np.float32)

    sample = ground[valid][:: max(1, int(valid.sum()) // 5000)]
    pixels = sample @ ground_to_image.T
    pixels /= pixels[:, 2:3]
    recovered = pixels @ image_to_ground.T
    recovered /= recovered[:, 2:3]
    ground_error_m = np.linalg.norm(recovered[:, :2] - sample[:, :2], axis=1)
    reprojection = recovered @ ground_to_image.T
    reprojection /= reprojection[:, 2:3]
    pixel_error = np.linalg.norm(reprojection[:, :2] - pixels[:, :2], axis=1)
    metrics = {
        "condition_number": float(np.linalg.cond(ground_to_image)),
        "valid_coverage_ratio": float(valid.mean()),
        "roundtrip_pixel_median": float(np.median(pixel_error)),
        "roundtrip_pixel_p95": float(np.percentile(pixel_error, 95)),
        "roundtrip_ground_median_m": float(np.median(ground_error_m)),
        "roundtrip_ground_p95_m": float(np.percentile(ground_error_m, 95)),
        "transform_contract": "manifest T_ego_cam values interpreted camera_to_ego; inverted to ego_to_camera",
    }
    return CameraGeometry(
        name=name,
        map_x=map_x.astype(np.float32),
        map_y=map_y.astype(np.float32),
        valid=valid,
        weight=weight,
        ground_to_image=ground_to_image,
        image_to_ground=image_to_ground,
        metrics=metrics,
    )


def remap_to_bev(image: np.ndarray, geometry: CameraGeometry, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    projected = cv2.remap(
        image.astype(np.float32),
        geometry.map_x,
        geometry.map_y,
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    projected[~geometry.valid] = 0
    return projected

