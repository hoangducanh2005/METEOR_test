"""CLI for the single-model Mask2Former drivable/lane pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from . import (
    mask2former_unified,
    unified_bev,
    unified_evaluation,
    unified_renderer,
    unified_validation,
)
from .constants import CAMERAS
from .io import atomic_json, load_task1_manifest
from .unified_config import load_unified_config


DEFAULT_SCENE = Path("out/bevlane/pnk_295_1772351165_1772351179")
DEFAULT_OUTPUT = Path("out/task2_lane_bev/pnk_295_1772351165_1772351179/mask2former_unified")
DEFAULT_CONFIG = Path("pnk_lane_bev/configs/segmentation/mapillary_swin_large.yaml")
DEFAULT_ALLROOT = Path("out/allroot/pnk_295_1772351165_1772351179/run_manifest.json")
DEFAULT_SOURCE_CACHE = Path("out/pnk_e2e/cache/panoptic")
DEFAULT_BASELINE = Path("out/task2_lane_bev/pnk_295_1772351165_1772351179")


def prepare(scene_dir: Path, output_dir: Path, config_path: Path) -> dict:
    manifest = load_task1_manifest(scene_dir)
    config = load_unified_config(config_path)
    for relative in (
        "cache/mask2former_unified", "bev", "bev_visual/per_camera", "metrics",
        "artifacts/per_camera", "artifacts/fused_bev", "deliverables/frames",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    resolved = output_dir / "config_resolved.yaml"
    resolved.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    run_manifest = {
        "schema_version": 1,
        "task": "unified_mask2former_drivable_lane_bev",
        "scene": manifest["scene"],
        "scene_dir": str(scene_dir.resolve()),
        "model": config["model"]["name"],
        "inference_path": [
            "multi_camera_rgb", "mask2former_mapillary_panoptic", "class_filter",
            "lane_postprocess", "calibrated_ipm", "confidence_weighted_fusion",
        ],
        "excluded_from_inference_path": ["UFLD-v2", "CLRNet"],
        "cameras": list(CAMERAS),
        "frames": len(manifest["frames"]),
        "camera_images": len(manifest["frames"]) * len(CAMERAS),
        "taxonomy": config["taxonomy"],
        "reference_policy": "SELF_SOURCE_COUPLED_TASK1_PROXY; human GT unavailable",
    }
    atomic_json(output_dir / "run_manifest.json", run_manifest)
    return run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One Mask2Former Mapillary model -> drivable/lane -> IPM -> six-camera BEV"
    )
    parser.add_argument(
        "command",
        choices=("prepare", "infer", "benchmark", "project", "evaluate", "render", "validate", "run"),
    )
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--allroot-manifest", default=str(DEFAULT_ALLROOT))
    parser.add_argument("--source-panoptic-cache", default=str(DEFAULT_SOURCE_CACHE))
    parser.add_argument("--baseline-output", default=str(DEFAULT_BASELINE))
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--benchmark-frames-per-camera", type=int, default=2)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scene = Path(args.scene).resolve()
    output = Path(args.output).resolve()
    config = Path(args.config).resolve()
    if args.command in ("prepare", "run"):
        summary = prepare(scene, output, config)
        print(f"[prepare-unified] frames={summary['frames']} camera_images={summary['camera_images']}")
    if args.command in ("infer", "run"):
        mask2former_unified.run_inference(
            scene, output, config, Path(args.allroot_manifest).resolve(),
            Path(args.source_panoptic_cache).resolve() if args.source_panoptic_cache else None,
            device=args.device, force=args.force, limit=args.limit,
        )
    if args.command in ("project", "run"):
        unified_bev.run_projection(scene, output, config, force=args.force, limit=args.limit)
    if args.command in ("benchmark", "run"):
        mask2former_unified.benchmark_inference(
            scene, output, config, device=args.device,
            frames_per_camera=args.benchmark_frames_per_camera,
        )
    if args.command in ("evaluate", "run"):
        summary = unified_evaluation.evaluate(
            scene, output, Path(args.baseline_output).resolve(), limit=args.limit
        )
        print(
            f"[evaluate-unified] lane_f1={summary['After']['lane_f1']:.4f} "
            f"empty={summary['After']['empty_prediction_rate']:.4f}"
        )
    if args.command in ("render", "run"):
        unified_renderer.render(scene, output, fps=args.fps, limit=args.limit)
    if args.command in ("validate", "run"):
        report = unified_validation.validate(scene, output, limit=args.limit)
        print(f"[validate-unified] status={report['status']} failures={len(report['failures'])}")


if __name__ == "__main__":
    main()
