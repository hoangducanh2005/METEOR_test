#!/usr/bin/env python3
"""Create explicit per-frame occupancy-flow labels from agent track futures.

Input: ``agent_traj/<frame>.npz`` created from the scene's 3D track boxes.
Output: ``occ_flow/<frame>.npz`` with ``flow[2,H,W]`` (m/s, forward/left)
and ``mask[H,W]`` marking agent-box cells supervised by the flow task.
This follows the same target construction used by the METEOR flow loss.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.extract_gt import OUT  # noqa: E402
from bevlane.flow_targets import build_flow_target  # noqa: E402


def _scenes(value: str | None) -> list[str]:
    if not value:
        return sorted(p.name for p in Path(OUT).iterdir()
                      if (p / "manifest.json").exists())
    path = Path(value)
    if path.is_file():
        return path.read_text(encoding="utf-8").split()
    return [part.strip() for part in value.split(",") if part.strip()]


def process_scene(scene: str, *, force: bool = False) -> dict:
    out_dir = Path(OUT) / scene
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest["frames"]
    out_flow = out_dir / "occ_flow"
    out_flow.mkdir(parents=True, exist_ok=True)

    flow_pixels = moving_pixels = moving_agents = stationary_agents = 0
    agent_total = 0
    frame_count = 0
    peak_speed = 0.0
    for frame in frames:
        fi = int(frame["frame"])
        rel = frame.get("agent_traj")
        if not rel:
            raise RuntimeError(f"{scene} frame {fi}: missing agent_traj reference")
        source_path = out_dir / rel
        if not source_path.exists():
            raise FileNotFoundError(f"Missing agent trajectory: {source_path}")

        target_path = out_flow / f"{fi:04d}.npz"
        if target_path.exists() and not force:
            with np.load(target_path) as saved:
                valid = np.asarray(saved["mask"], dtype=np.uint8)
                flow = np.asarray(saved["flow"], dtype=np.float32)
        else:
            with np.load(source_path) as source:
                boxes = np.asarray(source["boxes"], dtype=np.float32)
                count = int(source["count"])
                traj = np.asarray(source["traj"], dtype=np.float32)
                tvalid = np.asarray(source["tvalid"], dtype=np.float32)
            flow, valid = build_flow_target(boxes, count, traj, tvalid, 200, 200)
            np.savez_compressed(target_path, flow=flow, mask=valid)

        if not target_path.exists():
            raise RuntimeError(f"Failed to write {target_path}")
        frame["occ_flow"] = f"occ_flow/{fi:04d}.npz"
        frame_count += 1
        flow_pixels += int(valid.sum())
        speed = np.linalg.norm(flow, axis=0)
        moving = valid.astype(bool) & (speed > 0.5)
        moving_pixels += int(moving.sum())
        if moving.any():
            peak_speed = max(peak_speed, float(speed[moving].max()))

        # Track-level counts are taken from the input, not inferred from the
        # raster, so the summary also covers boxes with sub-pixel footprints.
        with np.load(source_path) as source:
            boxes = np.asarray(source["boxes"], dtype=np.float32)
            count = min(int(source["count"]), len(boxes))
            traj = np.asarray(source["traj"], dtype=np.float32)
            tvalid = np.asarray(source["tvalid"], dtype=np.float32)
        for k in range(count):
            _, xe, ye, length, _, _ = boxes[k]
            if length <= 0 or abs(xe) > 42 or abs(ye) > 42:
                continue
            agent_total += 1
            v = (traj[k, 0] / 0.5) if tvalid[k, 0] > 0.5 else np.zeros(2)
            if float(np.linalg.norm(v)) > 0.5:
                moving_agents += 1
            else:
                stationary_agents += 1

    meta = {
        "task": "occupancy_flow",
        "scene": scene,
        "frames": frame_count,
        "grid": {"shape": [200, 200], "resolution_m": 0.4,
                 "x_forward_range_m": [-40.0, 40.0],
                 "y_left_range_m": [-40.0, 40.0]},
        "flow": {"shape_per_frame": [2, 200, 200],
                 "channels": ["vx_forward_mps", "vy_left_mps"],
                 "target_horizon_s": 0.5,
                 "valid_mask": "1 on tracked-agent box cells; 0 elsewhere"},
        "source": "agent_traj boxes and first future displacement",
        "supervision_policy": "stationary or missing first future waypoint => zero velocity inside box",
        "agent_count_in_range": agent_total,
        "moving_agent_count_gt_0_5mps": moving_agents,
        "stationary_or_no_future_agent_count": stationary_agents,
        "valid_agent_box_pixels": flow_pixels,
        "moving_agent_pixels_gt_0_5mps": moving_pixels,
        "peak_labeled_speed_mps": peak_speed,
        "raw_data_modified": False,
    }
    manifest["occ_flow_meta"] = meta
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "occ_flow_summary.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None,
                    help="scene names, comma-separated, or path to scene list")
    ap.add_argument("--force", action="store_true",
                    help="regenerate existing occupancy-flow labels")
    args = ap.parse_args()
    scenes = _scenes(args.scenes)
    if not scenes:
        raise SystemExit("No scenes found")
    for scene in scenes:
        result = process_scene(scene, force=args.force)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
