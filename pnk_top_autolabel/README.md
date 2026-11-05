# PNK LIDAR_TOP → METEOR autolabel pipeline

This runner joins sensor preprocessing, segment-local heading/yaw resolution,
Mask2Former Swin-L image labels, the T4Dataset adapter, METEOR Autolabel Factory,
occupancy-flow export, and a one-sample training smoke run. CoMET source is not
available in this repository.

## Default run

```powershell
& .\pnk_top_autolabel\run_full.ps1
```

The default input is:

```text
D:\mentor_data\PNKData\20260301_1429_VF6_01_1772350177_1772350814
```

It uses the matching 3D reference-label segment
`290_1772350499_1772350516`, all eight calibrated camera streams available in
this recording, and `LIDAR_TOP` only. It does not modify raw sensors or labels.

## Stages

1. Deskew each TOP-LiDAR scan with per-point timestamps and NAV poses; audit
   camera synchronization/dropout/freeze.
2. Re-reference the nearest TOP scan to each labeled timestamp and build the
   T4Dataset scene.
3. Recompute each `Car`/`Rider` world trajectory in local ENU. Check the
   segment against the established `NAV Heading2 + raw yaw` convention, resolve
   front/back from motion, and propagate only within the same continuous track.
   Unresolved headings are marked invalid. The original labels stay read-only.
4. Run Mask2Former Swin-L Mapillary Vistas on each camera image to generate
   general 2D panoptic labels. Supplied PNK 3D boxes remain the 3D box source;
   this run does not run CenterPoint.
5. Convert through METEOR Autolabel Factory, export occupancy-flow targets from
   tracks with valid future displacement, render per-frame images/video, then
   run one training sample as a handoff smoke check.

## Outputs

- `out/pnk_top_e2e_290_1772350499_1772350516/preprocessing/`: deskewed TOP
  clouds, calibration visualizations, and sensor health reports.
- `out/allroot/pnk_top_290_1772350499_1772350516/annotation/`: T4 general
  annotations and run-local heading audit (`heading_candidate_comparison.csv`,
  `resolved_world_headings.csv`, and `annotation_heading_audit.json`).
- `out/bevlane/pnk_top_290_1772350499_1772350516/`: METEOR Factory output,
  occupancy-flow arrays, and flow summary.
- `out/pnk_top_e2e_290_1772350499_1772350516/reports/`: run report, checklogs,
  per-frame PNGs, and MP4.
- `out/pnk_top_e2e_290_1772350499_1772350516/train_smoke/`: one-sample training
  handoff output.

For another recording, pass its raw folder and matching label directory to the
same runner, and give it a unique `-Scene` and `-RunName`:

```powershell
& .\pnk_top_autolabel\run_full.ps1 `
  -Raw 'D:\mentor_data\PNKData\<session>' `
  -Labels 'D:\mentor_data\PNKData_meta_fresh\label\<segment>\label' `
  -Scene 'pnk_top_<segment>' -RunName 'pnk_top_e2e_<segment>'
```
