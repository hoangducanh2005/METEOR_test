"""CLI for the PNK LIDAR_TOP → METEOR E2E pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pnk_comet_autolabel.mask2former import DEFAULT_MODEL
from pnk_comet_autolabel.validator import validate_factory_segmentation, validate_scene

from .pipeline import (
    build_t4_scene,
    export_general_annotations,
    run_mask2former,
    run_sensor_preprocessing,
    write_run_report,
)


def emit(value) -> None:
    print(json.dumps(value, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("preprocess", help="deskew every LIDAR_TOP scan and audit camera health")
    prep.add_argument("--raw", type=Path, required=True)
    prep.add_argument("--calibration", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--visualization-stride", type=int, default=10)

    build = sub.add_parser("build-scene", help="re-reference deskewed TOP scans to annotation times")
    for name in ("raw", "calibration", "labels", "preprocessed", "preflight", "staging", "scene-dir"):
        build.add_argument("--" + name, type=Path, required=True)
    build.add_argument("--scene-name", required=True)
    build.add_argument("--vehicle", default="VF6_01")

    infer = sub.add_parser("infer-2d", help="run/reuse Mask2Former Swin-L panoptic predictions")
    infer.add_argument("--scene-dir", type=Path, required=True)
    infer.add_argument("--cache-dir", type=Path, required=True)
    infer.add_argument("--model", default=DEFAULT_MODEL)
    infer.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    infer.add_argument("--reuse-scene", type=Path)
    infer.add_argument("--reuse-cache", type=Path)
    infer.add_argument("--report", type=Path)
    infer.add_argument("--force", action="store_true")

    export = sub.add_parser("export", help="export general annotations and apply the heading audit sidecar")
    export.add_argument("--scene-dir", type=Path, required=True)
    export.add_argument("--cache-dir", type=Path, required=True)
    export.add_argument("--heading-audit", type=Path,
                        help="optional external audit; by default use the run-local audit from build-scene")
    export.add_argument("--segment", help="optional segment override")

    validate = sub.add_parser("validate", help="validate T4Dataset annotation contract")
    validate.add_argument("--scene-dir", type=Path, required=True)
    validate.add_argument("--report", type=Path, required=True)

    factory_validate = sub.add_parser("validate-factory", help="require seg2d21 for every camera view")
    factory_validate.add_argument("--scene-dir", type=Path, required=True)
    factory_validate.add_argument("--report", type=Path, required=True)

    report = sub.add_parser("report", help="write run summary after factory and training finish")
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--scene-name", required=True)
    report.add_argument("--preprocessing", type=Path, required=True)
    report.add_argument("--preflight", type=Path, required=True)
    report.add_argument("--scene-dir", type=Path, required=True)
    report.add_argument("--factory-report", type=Path)
    report.add_argument("--training-log", type=Path)

    args = parser.parse_args()
    if args.command == "preprocess":
        emit(run_sensor_preprocessing(args.raw, args.calibration, args.out,
                                      args.visualization_stride))
    elif args.command == "build-scene":
        emit(build_t4_scene(args.raw, args.calibration, args.labels, args.preprocessed,
                            args.preflight, args.staging, args.scene_dir, args.scene_name,
                            args.vehicle))
    elif args.command == "infer-2d":
        result = run_mask2former(args.scene_dir, args.cache_dir, args.model, args.device,
                                 args.force, args.reuse_scene, args.reuse_cache)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
        emit(result)
    elif args.command == "export":
        emit(export_general_annotations(args.scene_dir, args.cache_dir,
                                        args.heading_audit, args.segment))
    elif args.command == "validate":
        emit(validate_scene(args.scene_dir, args.report))
    elif args.command == "validate-factory":
        emit(validate_factory_segmentation(args.scene_dir, args.report))
    elif args.command == "report":
        emit(write_run_report(args.out, args.scene_name, args.preprocessing, args.preflight,
                              args.scene_dir, args.factory_report, args.training_log))


if __name__ == "__main__":
    main()
