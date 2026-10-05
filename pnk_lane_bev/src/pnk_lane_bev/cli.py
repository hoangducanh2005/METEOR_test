"""Command line interface for the Task 2 lane/drivable BEV pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from . import (
    lane_evaluation,
    lane_pipeline,
    lane_renderer,
    lane_validation,
    mask2former_small,
    metrics,
    project,
    renderer,
    ufld_v2,
    validator,
)
from .constants import MASK2FORMER_MODEL, UFLD_SOURCE_COMMIT, UFLD_SOURCE_REPOSITORY
from .io import atomic_json, prepare_run


DEFAULT_SCENE = Path("out/bevlane/pnk_295_1772351165_1772351179")
DEFAULT_OUTPUT = Path("out/task2_lane_bev/pnk_295_1772351165_1772351179")
DEFAULT_CHECKPOINT = Path("pnk_lane_bev/models/culane_res18.pth")
DEFAULT_UFLD34_CHECKPOINT = Path("pnk_lane_bev/models/culane_res34.pth")
DEFAULT_CLRNET_CHECKPOINT = Path("pnk_lane_bev/models/culane_r18.pth")
DEFAULT_CAMERA_CONFIG = Path("pnk_lane_bev/configs/lane/cameras/pnk_6cam.yaml")
DEFAULT_ALLROOT_MANIFEST = Path("out/allroot/pnk_295_1772351165_1772351179/run_manifest.json")


def _paths(args) -> tuple[Path, Path]:
    return Path(args.scene).resolve(), Path(args.output).resolve()


def _preflight(scene: Path, output: Path) -> dict:
    run_manifest = prepare_run(scene, output)
    config = {
        "input": {
            "scene": str(scene),
            "frames": run_manifest["counts"]["frames"],
            "cameras": run_manifest["cameras"],
        },
        "models": {
            "drivable": MASK2FORMER_MODEL,
            "lane": "UFLD-v2 CULane ResNet-18",
            "ufld_source": UFLD_SOURCE_REPOSITORY,
            "ufld_commit": UFLD_SOURCE_COMMIT,
        },
        "bev": run_manifest["bev"],
        "evaluation": {
            "reference_type": "PROXY_TASK1_AND_SELF_CONSISTENCY",
            "audit_gt_available": False,
            "status_on_success": "PASS_PROXY_ONLY",
        },
        "render": {"canvas_wh": [2560, 1440], "fps": 5.0, "bev_position": "right_full_height"},
    }
    (output / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    atomic_json(output / "preflight_summary.json", run_manifest["counts"])
    print(f"[preflight] {run_manifest['counts']['frames']} frames, {run_manifest['counts']['camera_images']} images")
    return run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 2: six-camera lane/drivable detection and fused BEV")
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "infer-drivable",
            "infer-lane",
            "project-bev",
            "evaluate",
            "render",
            "validate",
            "run",
            "lane-v2-infer",
            "lane-v2-fuse",
            "lane-v2-evaluate",
            "lane-v2-render",
            "lane-v2-validate",
            "lane-v2-run",
        ),
    )
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--ufld-checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--ufld34-checkpoint", default=str(DEFAULT_UFLD34_CHECKPOINT))
    parser.add_argument("--clrnet-checkpoint", default=str(DEFAULT_CLRNET_CHECKPOINT))
    parser.add_argument("--camera-config", default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument("--allroot-manifest", default=str(DEFAULT_ALLROOT_MANIFEST))
    parser.add_argument("--detectors", nargs="+", choices=lane_pipeline.DETECTORS, default=list(lane_pipeline.DETECTORS))
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scene, output = _paths(args)
    checkpoint = Path(args.checkpoint).resolve()
    if args.command in ("preflight", "run"):
        _preflight(scene, output)
    if args.command in ("infer-drivable", "run"):
        mask2former_small.run_scene(scene, output, device=args.device, force=args.force, limit=args.limit)
    if args.command in ("infer-lane", "run"):
        ufld_v2.run_scene(scene, output, checkpoint, device=args.device, force=args.force, limit=args.limit)
    if args.command in ("project-bev", "run"):
        project.run_scene(scene, output, force=args.force, limit=args.limit)
    if args.command in ("evaluate", "run"):
        metrics.evaluate(scene, output, limit=args.limit)
    if args.command in ("render", "run"):
        renderer.render(scene, output, fps=args.fps, limit=args.limit)
    if args.command in ("validate", "run"):
        report = validator.validate(scene, output, limit=args.limit)
        print(f"[validate] status={report['status']} frames={report['counts']['frames']}")
    detectors = tuple(args.detectors)
    checkpoints = {
        "ufld_v2": Path(args.ufld_checkpoint).resolve(),
        "ufld_v2_res34": Path(args.ufld34_checkpoint).resolve(),
        "clrnet": Path(args.clrnet_checkpoint).resolve(),
    }
    camera_config = Path(args.camera_config).resolve()
    if args.command in ("lane-v2-infer", "lane-v2-run"):
        lane_pipeline.run_inference(
            scene,
            output,
            Path(args.allroot_manifest).resolve(),
            camera_config,
            checkpoints,
            detectors=detectors,
            device=args.device,
            force=args.force,
            limit=args.limit,
        )
    if args.command in ("lane-v2-fuse", "lane-v2-run"):
        lane_pipeline.run_fusion(
            scene, output, camera_config, detectors=detectors, force=args.force, limit=args.limit
        )
    if args.command in ("lane-v2-evaluate", "lane-v2-run"):
        summary = lane_evaluation.evaluate(scene, output, detectors=detectors, limit=args.limit)
        print(f"[lane-v2-evaluate] selected={summary['Conclusion']['selected_detector_by_lane_f1_proxy']}")
    if args.command in ("lane-v2-render", "lane-v2-run"):
        lane_renderer.render(scene, output, detectors=detectors, fps=args.fps, limit=args.limit)
    if args.command in ("lane-v2-validate", "lane-v2-run"):
        report = lane_validation.validate(scene, output, detectors=detectors, limit=args.limit)
        print(f"[lane-v2-validate] status={report['status']} images={report['camera_images']}")


if __name__ == "__main__":
    main()
