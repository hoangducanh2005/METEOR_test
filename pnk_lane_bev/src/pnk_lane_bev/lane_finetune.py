"""Target-domain annotation adapter used before UFLD/CLRNet fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path

from .io import atomic_json


def prepare_target_domain(annotation_jsonl: Path, output_manifest: Path) -> dict:
    """Validate a camera-aware polyline JSONL and emit a reproducible manifest.

    Each row must contain image, camera, width, height and lanes. Each lane holds
    points as [[x,y], ...]. This neutral contract can then be converted by the
    upstream UFLD-v2 or CLRNet dataset adapter selected in the training config.
    """
    records = []
    with annotation_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"image", "camera", "width", "height", "lanes"}
            if not required.issubset(record):
                raise ValueError(f"Line {line_number} is missing {sorted(required - set(record))}")
            if not Path(record["image"]).exists():
                raise FileNotFoundError(f"Line {line_number} image does not exist: {record['image']}")
            for lane_index, lane in enumerate(record["lanes"]):
                points = lane.get("points", [])
                if len(points) < 2 or any(len(point) != 2 for point in points):
                    raise ValueError(f"Line {line_number}, lane {lane_index}: invalid points")
            records.append(record)
    summary = {
        "schema_version": 1,
        "source": str(annotation_jsonl.resolve()),
        "records": len(records),
        "cameras": sorted({record["camera"] for record in records}),
        "images": [record["image"] for record in records],
        "status": "READY_FOR_UFLD_OR_CLRNET_ADAPTER",
    }
    atomic_json(output_manifest, summary)
    return summary

