# PNKData Preprocessing and METEOR Autolabel Progress

## Scope

End-to-end run: session `20260301_1429_VF6_01_1772350177_1772350814`, label segment `290_1772350499_1772350516`. The run used `LIDAR_TOP`, eight pinhole cameras, Mask2Former Swin-L Mapillary Vistas, a T4Dataset adapter, and the METEOR Autolabel Factory. Four fisheye cameras were excluded. Raw sensor files and source labels were not modified.

## Dataset audit before preprocessing

- Metadata index: **1,924 rows**; **220** marked 3D-annotated; **1,703** rows with `num_boxes=0` marked unannotated. This zero does not prove that the scene contains no objects.
- Label inventory: **23,401 files**, **1,353,970 boxes**, **76 empty files**, **0 malformed rows**. Classes: Car 564,376; Rider 671,802; Pedestrian 117,792.
- Exact label/`LIDAR_MERGED_EGO` timestamp pairs: **1,768 across 22 sessions**; geometric checks sampled **120 frames**.
- Box convention evidence supports `dx=length`, `dy=width`, `dz=height`; median Car dimensions are **4.40 × 1.95 × 1.68 m**. Geometric-center Z has a **0.111 m** median ground-proxy error versus **0.807 m** for bottom-center. The ground proxy is not surveyed ground truth.
- Raw ego-frame yaw/motion comparison was poor: **36.20° median / 80.30° P90** over 234,261 track pairs. The best world-frame candidate is `NAV Heading2 + raw yaw`: **2.20° / 11.86°** median/P90 axis error, with **94.57%** within 25° over 10,807 moving-Car observations. These are modulo-180° axis checks, not front/back accuracy.
- Calibration review used `Calib_2`; **96 overlays** were inspected and **803 extreme projections** were skipped. Some projections remained offset, so calibration is not fully validated. The older `Calib` bundle is marked unreliable.

## Preprocessing completed

- Deskewed **170/170 `LIDAR_TOP` scans** using per-point timestamps, `Calib_2`, and interpolated NAV position/Heading2/pitch/roll; re-referenced the cloud to **86 annotation anchors**. TOP timestamp delta was **0 ms** for selected anchors.
- Camera timing: P95/max LiDAR-camera delta **2.02/31.94 ms** (35 ms limit). Of **1,360** scan-camera pairs, **879 (64.6%)** were usable and **481 (35.4%)** invalid/suspect. The session had **9 dropout intervals** and **1,450 freeze candidates**; among the 688 anchor views, **252** were flagged as suspected freeze and **436** were usable for cross-modal checks.
- Transformed tracked box centers to local ENU to estimate motion and resolve yaw versus yaw+180° when motion gates passed. The run applied **4,659** headings (2,704 motion-selected; 1,955 propagated), left **5,381 Car/Rider** boxes invalid, and did not infer front/back for **324 pedestrians**. A dedicated parked-state classifier is not yet implemented; speed/jitter can still create false moving anchors.

### Yaw and heading resolution

The audit supports raw box `yaw` as the box long-axis angle in ego coordinates; no ±90° offset is applied. With ego `+X` forward and `+Y` left, convert NAV compass heading to world/ENU yaw:

```text
psi_ego_world = wrap(90 degrees - Heading2)
theta_axis_world = wrap(psi_ego_world + yaw_raw)
```

Transform each tracked box center into ENU using the ego pose, then estimate its world-motion vector from neighboring track centers:

```text
v_world = (p_world[t2] - p_world[t1]) / (t2 - t1)
theta_motion = atan2(v_y, v_x)
```

For a moving anchor, compare the two possible directed headings and choose the one closer to motion:

```text
theta_heading = argmin(theta in {theta_axis_world, theta_axis_world + 180 degrees})
               angular_error(theta, theta_motion)
```

The current anchor gates are object speed **1.5–40 m/s**, ego turn **≤8°/s**, NAV heading standard deviation **≤2°**, and box-axis-to-motion error **≤25°** (axis error is modulo 180°). Propagate only the 180° branch within the same track across gaps **≤0.75 s**, retaining each frame's measured box axis. Tracks without a passing moving anchor stay `heading_invalid`. Valid derived world headings update only the generated T4 `sample_annotation`; raw `.txt` labels remain unchanged.

This is object-track motion compensation, separate from point-wise LiDAR deskew. The current implementation uses speed thresholds but has no independent parked-state classifier, so a parked object with noisy centers can still become a false moving anchor. The run's heading counts are therefore derived outputs, not verified front/back accuracy.

## Autolabel and Factory evidence

- Mask2Former Swin-L ran on **688/688 camera views**, producing **13,827 mapped panoptic segments**. METEOR segmentation validation decoded **197,113,738 pixels**, returned **0 errors**, and passed its completeness gate.
- T4 contract: **PASS** for 86 samples, 688 required segmented views, and 10,364 3D annotations.
- General annotations exported: `surface_ann` **4,695**; `object_ann` **9,132**; `sample_annotation` **10,364**. The 3D boxes are supplied PNK reference labels, not CenterPoint predictions.
- Occupancy flow was generated for **86 frames** on a 200×200 grid at 0.4 m with a 0.5 s horizon: **5,142** agents in range, **2,246** moving above 0.5 m/s, **2,896** stationary or without future displacement, and **88,325** supervised box cells. These are target-coverage counts, not flow accuracy metrics.
- Training smoke drew one sample for one optimizer step. Validation had **0 samples**; the log reports `mIoU=NaN` and `ADEc=1e9` (empty-validation sentinel), so it provides no model-quality result.

## METEOR task-output coverage

The Factory manifest contains artifacts for the following **12 task-output families**. This counts generated artifacts, not independently validated task accuracy. The repository's v26 architecture document uses a different count for neural-network heads (seven).

| # | Task output | Run status / provenance |
|---:|---|---|
| 1 | BEV lane/surface raster | Generated (`gt`) |
| 2 | Vector lane map | Generated (`gt_vec`) |
| 3 | Metric depth | Generated (`depth_gt4`, `depth_gt4n`) |
| 4 | 2D semantic segmentation | Generated; completeness/decode gate passed |
| 5 | 2D boxes | Generated (`bbox2d`) |
| 6 | 3D boxes | Generated from PNK reference annotations; no CenterPoint inference |
| 7 | Ego-motion/driving targets | Generated from poses |
| 8 | 3D occupancy | Generated (`occ`) |
| 9 | Agent trajectories | Generated from reference tracks |
| 10 | Occupancy flow | Generated from tracks and future displacement |
| 11 | Lane graph | Generated (`lanegraph.npz`) |
| 12 | Risk map | Generated (`risk_map.npz`) |

**Coverage: 12/12 artifact families generated for this run; 0/12 have independent PNK accuracy metrics.** Stop-line and dedicated traffic-light-state outputs are excluded by scope. A generic traffic-light class may appear in 2D segmentation; that is not a traffic-light-state label. `unknown_obj` is an additional auxiliary artifact outside this 12-item count.

## Evidence files

- [Run summary](../out/pnk_top_e2e_290_1772350499_1772350516/reports/run_summary.md)
- [Preprocessing report](../out/pnk_top_e2e_290_1772350499_1772350516/preprocessing/preprocessing_report.md)
- [T4 contract check](../out/pnk_top_e2e_290_1772350499_1772350516/reports/t4_contract.json)
- [Factory segmentation check](../out/pnk_top_e2e_290_1772350499_1772350516/reports/factory_segmentation.json)
- [Factory manifest](../out/bevlane/pnk_top_290_1772350499_1772350516/manifest.json)
- [Dataset box audit](../dataset_check/audit_summary.md) · [World-frame yaw audit](../dataset_check/world_yaw_audit/world_yaw_summary.md)
