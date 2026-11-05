# PNK 3D Ground Truth Reverse Audit

## Scope and coverage

- Raw sensor root: `D:\mentor_data\PNKData`
- Label root: `D:\mentor_data\PNKData_meta_fresh\label\2026_08_14_vf6_01_02`
- Metadata index: `D:\mentor_data\PNKData_meta_fresh\file_csv\2026_08_14_vf6_01_02.csv`
- Metadata rows: **1,924**; `3d_annotated=1`: **220**; `num_boxes=0`: **1,703** (zero rows are marked not annotated in this metadata index).
- Label text files: **23,401**; parsed box rows: **1,353,970**; empty files: **76**; malformed rows: **0**.
- Label class counts: **{'Pedestrian': 117792, 'Car': 564376, 'Rider': 671802}**.
- Exact timestamp pairs available: **1,768**; expensive geometry audit sampled **120** frames, stratified across sessions. See `raw_frame_sample.csv`.
- Point-cloud checks use `LIDAR_MERGED_EGO`; label and cloud are paired by exact timestamp stem. Results do not cover frames with no local raw LiDAR pair.

## Findings

- Car dimension medians: median dx=4.40m, dy=1.95m, dz=1.68m. Physical ranges support `dx=length, dy=width`: dx is within 3–6 m for 94.73471586318341% of cars; dy is within 1.5–2.5 m for 95.37914440018712%.
- Best-ranked tested candidate under the multi-evidence score: **yaw `yaw`, `dx=length,dy=width`, z `geometric_center`**. Ranking includes point-count agreement, dimension priors, PCA long-axis alignment, local ground contact, and motion consistency.
- Candidate Car boxes: 3398; point-count Pearson / Spearman: 0.9974710804870562 / 0.9999010828832373; MAE / median AE: 22.937021777516186 / 0.0 points; exact match: 61.56562683931725%.
- PCA long-axis median / P90: 9.293416753900212 / 48.34070355087585 deg (modulo 180; N=2483); below 5/10/20 deg: 30.89005235602094% / 52.47684252919855% / 79.37978252114378%.
- Z center comparison: geometric-center point-count median AE / exact match = 0.0 / 61.56562683931725%; bottom-center = 29.0 / 16.98057680988817%. Ground proxy median absolute error = 0.11109991464614855 m for geometric center vs 0.8070500098198652 m for bottom center.
- Camera projection: 96 overlays across 8 cameras / 12 timestamps; mean LiDAR-camera delta 1.1 ms. 803 extreme projections skipped. See `visualizations/camera_projection_manifest.csv`.
- Local ground proxy error for selected Z convention, median / P90: 0.11109991464614855 / 0.6459699737739566 m. This uses the 10th percentile of nearby low LiDAR returns, **not a surveyed ground plane**.
- Moving-car heading candidate `yaw` has orientation-mod-180 median / P90 36.20152846773723° / 80.29848473345677° across 234261 moving track pairs. Directional front/back is separately recorded in `track_yaw_candidate_summary.csv`.

## Interpretation / confidence

- **Dimensions — high confidence:** both population distributions and point-count agreement support `dx=length, dy=width`; swapping dimensions degrades point-count fit. `dz` is vertical extent.
- **Center convention - HIGH for Z:** geometric center is strongly supported by point-count and local-ground-proxy comparisons. `z` is likely the box geometric center; XY center offsets were not separately perturbed. The proxy is weak on occluded, sloped, or sparse areas.
- **Yaw axis/sign - MEDIUM; front/back - LOW:** `yaw` with direct dimensions ranks first, with PCA long-axis median error 9.293416753900212 deg. Under ego axes (+X forward, +Y left), this supports the long axis at `yaw` from +X, positive toward +Y. PCA cannot distinguish 180-degree reversal, and temporal motion does not resolve front/back reliably.
- **Coordinate handedness - LOW / unresolved:** yaw candidates and track motion were compared, but motion errors remain broad. Camera overlays are qualitative and do not establish handedness by themselves.
- **Track continuity:** flagged anomaly counts are `{'dimension_outlier': 22264, 'direct_box_pointcount_mismatch': 28, 'track_yaw_flip_near_180deg': 698, 'track_dimension_range': 126, 'track_dimension_jump': 218, 'track_teleport': 265, 'track_speed_over_50mps': 264}`. These heuristic alerts need review; they are not confirmed annotation errors.
- **Point coverage:** `point_coverage_validation.csv` records local-neighborhood occupancy, point-to-box-surface distances, and per-axis quartile occupancy. Nearby non-object returns can enter the denominator, so treat it as diagnostic rather than semantic object recall.
- **`numpoints`:** correlation/exactness compare GT with points in the timestamp-matched merged cloud. A mismatch can mean different filtering/accumulation or sensor source, not automatically incorrect labels.
- **Camera cross-check:** date-matched `Calib_2` was used with nearest camera frames within 120 ms. Extreme projected boxes are skipped; final overlays are in `visualizations/camera_projection_final/`. Some boxes align plausibly, but others remain offset or produce long edges, so this does not pass as proof of calibration alignment. Colors: Car=orange, Rider=cyan, Pedestrian=magenta; numbers are local label-row indices, not track IDs.
- No source labels were modified. All computed results and visualizations are under `D:\GIT\METEOR_test\dataset_check`.

## Artifacts

- `dimension_stats.csv`
- `convention_comparison.csv`
- `pca_yaw_error.csv`
- `pointcount_validation.csv`
- `point_coverage_validation.csv`
- `track_motion_validation.csv`, `track_yaw_candidate_summary.csv`
- `anomalies.csv`, `dimension_outliers.csv`
- `raw_frame_sample.csv`
- `visualizations/camera_projection_manifest.csv`
- `visualizations/histograms/`, `visualizations/bev_samples/`, and `visualizations/camera_projection_final/`

## Conversion candidate

For a reversible converter prototype, use the top-ranked row in `convention_comparison.csv` (`yaw`, `dx=length,dy=width`, `geometric_center`). Keep it configurable and verify camera overlays plus downstream METEOR schema expectations before bulk conversion; do not mutate source `.txt` labels in place.

## World-frame heading follow-up (read-only)

- NAV pose was available for all **1,768 / 1,768** exact label–`LIDAR_MERGED_EGO` timestamp pairs. A separate audit compares NAV `Heading` and `Heading2` after ego-motion compensation; candidate metrics use the same **10,807** moving Car observations for both fields.
- Best tested combination: **NAV `Heading2` + raw box `yaw`**. World motion-axis error is median/P90 **2.20° / 11.86°**, with **94.57%** of observations within 25°. `Heading` + `yaw` on the same observations gives **4.58° / 25.81°**, with **89.70%** within 25°.
- This supports the box's modulo-180 axis and gives evidence for resolving front/back on moving tracks under a forward-motion assumption. The derived table has **10,601** direct-motion headings, **7,241** track-propagated headings, and **32,481** `heading_invalid` rows; keep invalid headings unresolved.
- This is a derived candidate, not a bulk label conversion. NAV field semantics and GNSS-to-LiDAR lever arm are unverified; stationary-only tracks cannot reveal front/back from LiDAR motion. Point-cloud deskew was not applied.
- Full methods, per-candidate comparison, NAV sync/quality fields, and derived headings are in [`world_yaw_audit/world_yaw_summary.md`](world_yaw_audit/world_yaw_summary.md), [`world_yaw_audit/world_yaw_candidate_comparison.csv`](world_yaw_audit/world_yaw_candidate_comparison.csv), and [`world_yaw_audit/resolved_world_headings.csv`](world_yaw_audit/resolved_world_headings.csv). All outputs remain under `dataset_check`; raw data and source labels were not modified.
