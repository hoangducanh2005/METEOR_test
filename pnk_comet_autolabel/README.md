# PNK CoMET replacement pilot

This package implements the runnable pilot described in
[`docs/COMET_REPLACEMENT_IMPLEMENTATION_PLAN.md`](../docs/COMET_REPLACEMENT_IMPLEMENTATION_PLAN.md):

```text
PNK raw -> Mask2Former + T4 adapter -> general annotations
        -> METEOR Autolabel Factory -> METEOR train smoke -> PNG/MP4
```

The pilot uses all 71 `LIDAR_MERGED_EGO` keyframes and their nearest images
from the eight pinhole cameras. The four fisheye cameras are outside this test.

## Environment

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv sync --project pnk_comet_autolabel --extra mask2former

# Replace the default CPU wheel with CUDA on the RTX 4050 host.
uv pip install --python pnk_comet_autolabel\.venv\Scripts\python.exe `
  --reinstall torch==2.7.1 torchvision==0.22.1 `
  --index-url https://download.pytorch.org/whl/cu128

uv pip install --python pnk_comet_autolabel\.venv\Scripts\python.exe `
  'scikit-image>=0.24'
```

## Full pilot

```powershell
.\pnk_comet_autolabel\run_pilot.ps1
```

Add `-ForceSegmentation` to recompute every Mask2Former cache. Without it, the
runner resumes from per-image caches under `out/pnk_e2e/cache/panoptic`.

The segmentation gates run twice: once against T4 `surface_ann`/`object_ann`
RLE and once against METEOR `seg2d21`. Rendering fails if any of the 568 camera
views lacks valid segmentation.

## 3D provenance

The current Windows pilot uses the 9,038 supplied 3D boxes as an explicitly
marked `reference_3d_bridge_for_e2e_flow_test`. This proves schema and
downstream flow but is not a CenterPoint prediction or a CenterPoint metric.
Replace `export_reference_boxes()` with the MMDetection3D CenterPoint backend
before using this pipeline to measure 3D autolabel quality.

