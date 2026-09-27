"""Command line interface for the PNK CoMET replacement pilot."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .exporter import export_panoptic, export_reference_boxes, prepare_scene
from .mask2former import DEFAULT_MODEL, run_scene
from .reader import write_preflight
from .report import write_run_summary
from .validator import validate_factory_segmentation, validate_scene


def emit(value) -> None:
    print(json.dumps(value, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)

    preflight = subs.add_parser("preflight", help="scan and timestamp-match a PNK raw folder")
    preflight.add_argument("--raw", type=Path, required=True)
    preflight.add_argument("--out", type=Path, required=True)
    preflight.add_argument("--max-delta-ms", type=float, default=35.0)

    prepare = subs.add_parser("prepare", help="write T4Dataset core records and converted sensor files")
    prepare.add_argument("--raw", type=Path, required=True)
    prepare.add_argument("--preflight", type=Path, required=True)
    prepare.add_argument("--calibration", type=Path, required=True)
    prepare.add_argument("--labels", type=Path, required=True)
    prepare.add_argument("--scene-dir", type=Path, required=True)
    prepare.add_argument("--scene-name", required=True)
    prepare.add_argument("--vehicle", default="VF6_01")
    prepare.add_argument("--recording-date", type=date.fromisoformat, default=date(2026, 3, 1))
    prepare.add_argument("--limit", type=int)

    infer = subs.add_parser("infer-2d", help="run Mask2Former and cache panoptic predictions")
    infer.add_argument("--scene-dir", type=Path, required=True)
    infer.add_argument("--cache-dir", type=Path, required=True)
    infer.add_argument("--model", default=DEFAULT_MODEL)
    infer.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    infer.add_argument("--limit-images", type=int)
    infer.add_argument("--force", action="store_true", help="overwrite existing per-image caches")

    export = subs.add_parser("export", help="export panoptic masks and the explicit pilot 3D bridge")
    export.add_argument("--scene-dir", type=Path, required=True)
    export.add_argument("--cache-dir", type=Path, required=True)
    export.add_argument("--boxes-source", choices=["reference", "none"], default="reference")

    validate = subs.add_parser("validate", help="validate the T4 annotation contract")
    validate.add_argument("--scene-dir", type=Path, required=True)
    validate.add_argument("--report", type=Path)

    factory = subs.add_parser("validate-factory-seg", help="enforce seg2d21 for every output view")
    factory.add_argument("--scene-dir", type=Path, required=True)
    factory.add_argument("--report", type=Path)

    report = subs.add_parser("report", help="summarize completed pilot artifacts")
    report.add_argument("--repo", type=Path, required=True)
    report.add_argument("--scene", required=True)
    report.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "preflight":
        emit(write_preflight(args.raw, args.out, args.max_delta_ms))
    elif args.command == "prepare":
        emit(prepare_scene(args.raw, args.preflight, args.calibration, args.labels,
                           args.scene_dir, args.scene_name, args.vehicle,
                           args.recording_date, args.limit))
    elif args.command == "infer-2d":
        emit(run_scene(args.scene_dir, args.cache_dir, args.model, args.device,
                       args.limit_images, args.force))
    elif args.command == "export":
        result = {"panoptic": export_panoptic(args.scene_dir, args.cache_dir)}
        result["boxes_3d"] = (export_reference_boxes(args.scene_dir)
                              if args.boxes_source == "reference" else {"source": "none"})
        emit(result)
    elif args.command == "validate":
        emit(validate_scene(args.scene_dir, args.report))
    elif args.command == "validate-factory-seg":
        emit(validate_factory_segmentation(args.scene_dir, args.report))
    elif args.command == "report":
        emit(write_run_summary(args.repo.resolve(), args.scene, args.out))


if __name__ == "__main__":
    main()
