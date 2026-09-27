"""Create a machine-readable summary from completed pilot artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_run_summary(repo: Path, scene: str, output: Path) -> dict:
    import cv2
    import torch

    reports = repo / "out" / "pnk_e2e" / "reports"
    preflight = _load(reports / "preflight.json")
    contract = _load(reports / "t4_contract.json")
    factory = _load(reports / "factory_segmentation.json")
    ann = repo / "out" / "allroot" / scene / "annotation"
    surface = _load(ann / "surface_ann.json")
    objects = _load(ann / "object_ann.json")
    boxes = _load(ann / "sample_annotation.json")
    instances = _load(ann / "instance.json")

    video_path = reports / "pnk_295_gt_full.mp4"
    cap = cv2.VideoCapture(str(video_path))
    video = {
        "png_frames": len(list((reports / "frames").glob("*.png"))),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "video_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "video_fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "video_bytes": video_path.stat().st_size,
        "video": video_path.name,
        "frames_dir": "frames",
    }
    cap.release()

    checkpoint = repo / "out" / "pnk_e2e" / "train_smoke" / "last.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved.get("model", saved.get("state_dict", {}))
    saved_args = saved.get("args", {})
    model_name = (saved_args.get("model", "unknown") if isinstance(saved_args, dict)
                  else getattr(saved_args, "model", "unknown"))
    finite = all(bool(torch.isfinite(value).all()) for value in state.values()
                 if torch.is_tensor(value))
    result = {
        "status": "PASS_WITH_3D_REFERENCE_BRIDGE",
        "scene": scene,
        "preflight": {
            "lidar_keyframes": preflight["lidar_input_count"],
            "camera_views": contract["required_camera_views"],
            "dropped_samples": preflight["dropped_count"],
            "max_camera_lidar_delta_ms": preflight["sync"]["max_delta_ms"],
            "p95_camera_lidar_delta_ms": preflight["sync"]["p95_delta_ms"],
        },
        "general_annotations": {
            "segmented_camera_views": contract["segmented_camera_views"],
            "panoptic_annotations": contract["segmentation_annotations"],
            "surface_ann": len(surface),
            "object_ann": len(objects),
            "sample_annotation": len(boxes),
            "instances_3d": len(instances),
            "model_2d": "facebook/mask2former-swin-large-mapillary-vistas-panoptic",
            "source_3d": "reference_3d_bridge_for_e2e_flow_test",
            "centerpoint_ran": False,
        },
        "factory": {
            "status": factory["status"],
            "frames": factory["frames"],
            "seg2d21_views": factory["segmented_views"],
            "segmentation_gate": factory["status"],
        },
        "deliverable": video,
        "train_smoke": {
            "status": "PASS" if finite else "FAIL",
            "model": model_name,
            "optimizer_steps": 1,
            "checkpoint": "../train_smoke/last.pt",
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": digest,
            "checkpoint_tensors": len(state),
            "checkpoint_all_finite": finite,
            "validation_metrics": None,
            "validation_note": "Single-scene flow test has no independent validation split.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
