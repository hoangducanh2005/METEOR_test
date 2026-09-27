"""Map pretrained panoptic classes into the METEOR taxonomy."""

from __future__ import annotations

import re

from .constants import MAPILLARY_TO_METEOR


def normalise_label(value: str) -> str:
    value = value.replace("--", " ").replace("/", " ").replace("_", " ")
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def map_mapillary_label(label: str) -> str | None:
    key = normalise_label(label)
    if key in MAPILLARY_TO_METEOR:
        return MAPILLARY_TO_METEOR[key]
    # Transformers checkpoints have used both hierarchical and short labels.
    # Suffix matching keeps the map stable across those metadata variants.
    for source, target in MAPILLARY_TO_METEOR.items():
        if key.endswith(source):
            return target
    return None

