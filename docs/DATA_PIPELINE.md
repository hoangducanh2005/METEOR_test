# The Autolabel Factory

Every METEOR supervision signal is generated automatically from t4dataset-format
recordings (NuScenes-like: `sample_data`, `ego_pose`, `calibrated_sensor`,
LiDAR sweeps, 2D panoptic `surface_ann`/`object_ann` as base64 COCO RLE,
3D `sample_annotation`). **No human labels are used.**

`bevlane/convert_dtset.py` drives all stages per scene (scene-parallel,
every stage skips existing outputs → fully resumable):

| # | Stage | Output |
|---|---|---|
| 1 | `run_batch.py` (autolabel production) | map-frame BEV class counts |
| 2 | `vectorize_bev.py` | connected lane polylines (`vector_map.json`) |
| 3 | `extract_gt.py` | cached 768×432 images + per-frame BEV crops |
| 4 | `add_narrow_cams.py` | tele-camera cache (mixed resolutions) |
| 5 | `render_vector_gt.py` | hybrid raster+vector lane GT (`gt_vec`) |
| 6 | `extract_depth_dense.py` / `_narrow.py` | dense metric depth, 8 cams |
| 7 | `extract_bev_box.py` | camera-confirmed oriented 3D boxes |
| 8 | `annotate_gtcov.py` | per-frame GT-coverage stats (training filter) |
| 9 | `extract_seg2d.py` | 21-class 2D segmentation (`seg2d21`) |
| 10 | `extract_bbox2d.py` + `extract_ego.py` + `extract_occ.py` | 2D boxes, E2E, occupancy |
| 11 | `extract_agent_traj.py` + `extract_occ_flow.py` | Agent futures and dense per-box occupancy flow |

## BEV lanes (`gt_vec`)

1. **Point labeling** — every LiDAR keyframe is projected into all cameras
   (pinhole + equidistant fisheye), sampling road-surface classes from the
   panoptic masks. Thin classes (lane/stop/edge/marking) only within 15 m
   (pixel→ground error grows with range).
2. **Map-frame accumulation** — ego_pose transforms points to the world frame;
   class hit-counts accumulate on a 0.1 m grid across the whole scene.
3. **Rasterisation** — area classes by argmax; thin classes by hit-count
   priority override; a `min_dist ≤ 20 m` masked variant kills far smear.
4. **Vectorisation** — skeletonise → trace → spur-prune → Douglas-Peucker →
   dash-gap linking (8 m for lane lines) → `vector_map.json` polylines in
   map-frame metres.
5. **Hybrid re-render** — per-frame ego crops (±80 × ±50 m @ 0.2 m): area
   classes from the raster, lines re-drawn from vectors at fixed width;
   enclosed road holes ≤ 400 cells filled; opposing carriageways removed by an
   ego-connected component filter **with a 30 % revert guard** (wide
   intersections would otherwise be wiped); `road_edge` recomputed as the
   pixel-accurate drivable boundary.

## Metric depth

LiDAR splat (min-depth per pixel) → `griddata` linear interpolation over the
**merged road+paint panoptic segment** (treating paint separately leaves holes
around lane lines) → geometric ground-plane fill for downward rays within 50°
(fisheye-pinhole guard) → same-segment EDT fill → segment median → sky 79.5 m,
ego 2.0 m. Stride-4 (108×192), 64 bins × 1.25 m.

## 2D segmentation (21 classes)

Taxonomy is **CSV-driven** (`comlops-21cls-autolabel-2504.csv`: id, name, RGB —
Cityscapes-like palette). Key recipe points:

- Thin classes (lane, marking, pole, sign, light) survive the 4× downsample via
  **coverage-based rasterisation**: a cell becomes the thin class if >12 % of
  its footprint is covered (naive nearest resize deletes sub-pixel lane lines;
  this recovered +49 % lane pixels).
- `ego_vehicle` is supervised as background 0 (ignore left the hood
  unconstrained → noise).
- All 8 cameras are labeled; 255 = ignore only where annotations are absent.

## 2D boxes (10 classes)

Taxonomy from `comlops-instance-2510.csv`. `object_ann` boxes scaled into the
768×432 cache, area-sorted, **KMAX = 96 per camera** (a KMAX of 32 silently
dropped 29 % of the annotations in crowded scenes — exactly the small objects:
cones, traffic lights, distant unknowns).

## 3D boxes (camera-confirmed)

LiDAR `sample_annotation` boxes are kept only when **geometrically confirmed by
a camera**: project the 8 corners into each camera and require overlap with a
same-class 2D annotation (inter / min-area > 0.3). 2D and 3D instance tokens
are disjoint in this dataset, so the match must be geometric. Occluded objects
never enter the BEV GT.

## E2E driving GT (no CAN needed)

From `ego_pose` alone (`extract_ego.py`):

- **Trajectory** — future poses at +0.5…+3.0 s transformed into the current
  ego frame (6 waypoints).
- **Speed / accel** — smoothed (±0.5 s) finite differences. Written for
  *every* frame (instantaneous signals need no future — a scene-tail v0 of 0
  once made the model predict "hold" at 80 km/h).
- **Steering** — bicycle model `atan(wheelbase · yaw_rate / v)`, masked below
  0.5 m/s.
- **Brake** — accel < −0.5 m/s².
- **valid = 0** where < 3 s of future exists (scene tails) — those frames
  still train all other tasks; the E2E loss is masked.

## 3D occupancy

Ego-frame voxels `[16, 200, 200]` @ 0.4 m (±40 m, z ∈ [−1, 5.4) m), classes:
free / obstacle / vehicle / two-wheeler / pedestrian / road / sidewalk /
vegetation / building / pole+sign, 255 = never observed.

- Points are labeled by sampling the **cached seg2d21 maps** (no RLE decode —
  a panoptic-decode variant cost 690 s/scene; this one ~45 s).
- ±8 strided frames accumulate via ego_pose; **dynamic classes come from ±1
  frames only** (sub-voxel smear, 3× the points of a single sweep — full
  accumulation drags moving cars across the map).
- **Free space** is carved by ray-stepping the current sweep at 0.4 m steps
  (coarser steps leave most above-road voxels unknown, which lets the model
  hallucinate structure there unpunished).

## Occupancy flow

`extract_occ_flow.py` converts the first future displacement in `agent_traj`
into an ego-frame velocity field on the 0.4 m BEV grid. Each tracked box's
oriented footprint receives `(vx, vy) = displacement / 0.5 s`; its mask marks
the cells supervised by flow loss. Stationary tracks or tracks without a
valid first future waypoint receive zero velocity on their box footprint.
This is the same target rasterizer used by `DepthSegIPMNet.flow_loss`, so the
saved artifact and train-time loss use identical geometry. The target inherits
the provenance and quality limits of the input 3D annotations and tracks.

## Quality gates (training-time)

- **Scene trimming** — first 3 and last 10 frames dropped (weak forward GT).
- **GT coverage filter** — per-frame labeled fractions (`gtcov`): core band
  ±30 m ≥ 3 %, forward band 30–80 m ≥ 0.5 %. Long-stationary spots (no
  accumulated GT ahead) are excluded automatically.
- **Scene selection** — indoor/underground scenes are detected by panoptic
  `sky` counts (NDT pose drift makes speed-based filtering unreliable);
  pose-jump frames are discarded.

## Data cleansing (ego localization & near-field road)

Cross-vehicle recordings need three independent checks before/after conversion
(portable to any t4dataset batch — thresholds derive from pose rate + sensor
geometry, not from a specific dataset):

- **Ego-pose yaw convention** (`bevlane/clean_ego_pose.py`) — some recordings
  express the ego forward axis along body +y instead of +x, so the pose yaw sits a
  constant 90° off the true heading; left uncorrected the map accumulation and every
  ego-frame crop are rotated 90°. Detected by the median heading-vs-motion error and
  fixed **per scene** (measured offset rounded to a multiple of 90°; a blanket
  correction breaks scenes that were already right). Non-destructive: writes a clean
  mirror and repoints the read root.
- **Broken localization** (`bevlane/scan_pose_quality.py`, reads the *corrected*
  poses) — **teleport**: consecutive high-rate pose steps > 8 m (NDT/GNSS jumps) →
  smeared, unusable map → excluded; **drift**: median heading-vs-motion error > 20°
  with no jump → mild lane-line ghosting → watch-list, usually kept. Detect jumps by
  step *distance*, not speed (near-duplicate timestamps give false speed spikes).
- **Near-field road holes** (fill in `autolabel_bev.process_scene`, `EGO_FILL_R`;
  post-process `bevlane/fill_ego_path.py`) — near the ego the LiDAR is blind
  (< ~1.4 m) and the ground just beyond projects onto the hood / into the nadir gap
  between the outward cameras, so the driven strip gets no panoptic label. Motion
  backfills it from earlier distant views; where the ego dwells (lights, congestion,
  parking) it stays empty. The ego is by definition on drivable surface → **stamp
  ROAD into UNLABELED cells within half a lane of the trajectory** (never overwrites
  an observed class). Scenes whose ego never travels enough (< ~10 m) to form a road
  corridor are excluded (BEV is only a near-field blob).

## PNKData preprocessing note: 3D box heading

Apply this policy before treating PNK `sample_annotation.rotation` as a front/back-correct vehicle heading:

- Use NAV `Heading2` + raw box `yaw` as the current best tested candidate, and transform timestamp-matched box centers into a world/local-ENU frame before estimating track motion. The audit compared both NAV heading fields on the same **10,807** moving Car observations: `Heading2 + yaw` had median/P90 motion-axis error **2.20° / 11.86°** (94.57% within 25°), versus **4.58° / 25.81°** for `Heading + yaw` (89.70% within 25°).
- Resolve the 180° front/back branch directly only when the motion axis is within 25° of the box axis, NAV quality and motion gates pass, and the forward-driving assumption is reasonable. Propagate only that 180° branch within the same track across gaps no larger than 0.75 s; retain each frame's measured box axis.
- Save derived heading separately with `heading_valid`, `heading_source`, and `heading_confidence`. If a track has no reliable moving anchor, retain `heading_valid=false` / `heading_invalid`; raw yaw may still describe its orientation modulo 180°. Do not invent a front/back direction for stationary tracks.
- In the audit subset, 10,601 rows were resolved directly from motion, 7,241 by within-track propagation, and 32,481 remained invalid. Recompute these counts for each preprocessing run; they are not universal dataset proportions.
- Keep original label files unchanged. Write converted annotations and derived heading metadata only into the generated/staging dataset. Do not apply a second point-cloud deskew until the `LIDAR_MERGED_EGO` deskew convention and sensor lever arm are verified.

The current PNK `export_reference_boxes()` bridge still exports the raw box yaw; it does not yet consume the derived heading audit. Until that is integrated and validated, treat its box orientation as modulo-180 where heading evidence is missing. See `dataset_check/world_yaw_audit/world_yaw_summary.md` for the audit and its limitations.

### Read-only LIDAR_TOP preprocessing command

The calibration/synchronization preprocessing entry point uses `LIDAR_TOP`, not `LIDAR_MERGED_EGO`. It writes deskewed cloud files, camera-health tables, and raw/undistorted projection overlays to an output directory outside the raw session:

```powershell
$session = '20260228_1331_VF6_01_1772260318_1772261864'
$env:PYTHONPATH = 'D:\GIT\METEOR_test\pnk_comet_autolabel\src'
python -m pnk_comet_autolabel.sensor_preprocessing `
  --raw "D:\mentor_data\PNKData\$session" `
  --calibration 'D:\mentor_data\PNKData_meta_fresh\Calib_2' `
  --out "D:\GIT\METEOR_test\dataset_check\top_preprocessing\$session" `
  --limit 20
```

Remove `--limit 20` to process the complete session. `camera_sync.csv` records per-camera timestamp delta, image reuse, freeze suspicion, and pose validity; `camera_dropout_gaps.csv` and `camera_freeze_candidates.csv` retain the detected intervals. `lidar_top_deskewed_ego/` contains clouds deskewed to each LiDAR scan-end pose, while `visualizations/` contains sampled projections onto raw and undistorted images.

The deskew is NAV-pose-based SE(3), using `Heading2`, `Pitch`, `Roll`, and `point_timestamp`; it assumes static scene points. Review `preprocessing_report.md` before accepting the calibration. The GNSS-to-ego lever arm is not applied, and dynamic objects are not deskewed by their own motion. The command is non-destructive and refuses an output directory inside the raw session or its parent folders.
