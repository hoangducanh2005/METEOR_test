"""Shared occupancy-flow target rasterizer.

Flow is the per-cell ego-frame velocity of the tracked agent whose current
oriented box covers that cell. The first agent-trajectory waypoint is at 0.5 s.
"""
from __future__ import annotations

import cv2
import numpy as np


def build_flow_target(boxes: np.ndarray, nbox: int, traj: np.ndarray,
                      tvalid: np.ndarray, height: int, width: int,
                      *, half_extent_m: float = 40.0,
                      resolution_m: float = 0.4,
                      horizon_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize tracked boxes as flow [forward,left] in m/s plus valid mask.

    Boxes use [class, x_forward, y_left, length, width, yaw] in the current
    ego frame. Tracks without a valid first future waypoint are supervised as
    stationary, matching ``DepthSegIPMNet.flow_loss``.
    """
    flow = np.zeros((2, height, width), dtype=np.float32)
    valid = np.zeros((height, width), dtype=np.uint8)
    count = min(max(int(nbox), 0), len(boxes))

    for k in range(count):
        cls, xe, ye, length, box_width, yaw = map(float, boxes[k])
        if length <= 0 or abs(xe) > half_extent_m + 2 or abs(ye) > half_extent_m + 2:
            continue

        vx = vy = 0.0
        if tvalid[k, 0] > 0.5:
            vx = float(traj[k, 0, 0]) / horizon_s
            vy = float(traj[k, 0, 1]) / horizon_s

        c, s = np.cos(yaw), np.sin(yaw)
        polygon = []
        for lx, wy in ((length / 2, box_width / 2),
                       (length / 2, -box_width / 2),
                       (-length / 2, -box_width / 2),
                       (-length / 2, box_width / 2)):
            px = xe + lx * c - wy * s
            py = ye + lx * s + wy * c
            col = int((half_extent_m - py) / resolution_m)
            row = int((half_extent_m - px) / resolution_m)
            polygon.append([col, row])

        instance_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(instance_mask,
                     [np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)], 1)
        covered = instance_mask > 0
        flow[0, covered] = vx
        flow[1, covered] = vy
        valid[covered] = 1

    return flow, valid
