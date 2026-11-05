# World-frame yaw and heading audit

## Scope

- Raw and labels are read-only. All derived artifacts are written in `D:\GIT\METEOR_test\dataset_check\world_yaw_audit`.
- Exact-timestamp matched label / `LIDAR_MERGED_EGO` frames: **1,768** across **22 sessions**.
- NAV interpolated successfully for **1,768** frames; rejected for NAV coverage/gap: **0**.
- Parsed boxes in those frames: **160,245**; Car track observations were evaluated under both NAV `Heading` and `Heading2`.
- NAV timestamps were interpolated only when bracketing samples are no more than 100 ms apart. `nav_lidar_sync.csv` contains per-frame deltas and NAV quality fields (`Status`, `Age`, `Warning`) for review; status codes are not interpreted here.
- Example point-time audit: `{'sample_laz': '1772260636-099738359.laz', 'point_timestamp_type': 'uint64', 'point_count': 280311, 'scan_offset_min_ms': -100.00061988830566, 'scan_offset_median_ms': -53.310394287109375, 'scan_offset_max_ms': 0.0}`. Point timestamps are recorded but this audit does **not** deskew or rewrite any cloud.

## Method

1. Interpolate the timestamp-matched ego pose from `OTHERS/NAV`.
2. Convert GPS lat/lon/alt to session-local ENU and transform each ego-frame box center into ENU. NAV compass heading is converted to ENU yaw as `90 deg - heading`, consistent with ego `+X` forward and `+Y` left.
3. Compare `Heading` and `Heading2` on common support: both heading values and both standard deviations must be present, both standard deviations at most 2 deg, and each observation must pass the speed (1.5-40 m/s) and ego-turn (<8 deg/s) gates under both NAV fields. Then evaluate all six raw-yaw transforms modulo 180 degrees.
4. For the selected candidate, resolve the 180-degree branch only when motion is within 25 deg of the box axis. Propagate that branch through contiguous observations of the same track, preserving each frame's measured box axis. Tracks without a confident moving anchor remain `heading_invalid`.

## Results

- NAV `Heading` vs compass course from `Ve/Vn` in the first session: median error **0.025751072073035175 deg**, P90 **0.057365404750595414 deg** over 3335 rows with ego speed >=2 m/s in the loaded audit window. See `nav_heading_course_check.csv` for both heading fields and all loaded windows.
- Best-ranked world candidate: **NAV `Heading2` + box yaw `yaw`**.
- Heading-field comparison uses the same **10,807** moving observations for every NAV/yaw candidate. Best candidate results: median/P90 axis error **2.1982768546978573 / 11.86087668039765 deg**; within 20/25 deg: **93.72628851670214% / 94.56833533820672%**.
- Low-motion tracks with sufficient ego rotation for a static-axis check: **529**; median per-track world-axis jitter **0.1736687114400733 deg**. This can help distinguish yaw sign without assuming a target car's velocity is always its forward direction.
- Resolved output rows: **50,323**; heading valid by direct world motion: **10,601**; propagated within track: **7,241**; still invalid: **32,481**.
- Previous ego-frame track baseline for raw `yaw`: orientation-mod-180 median/P90 **36.20152846773723 / 80.29848473345677 deg** over 234261 pairs. Compare with world-frame candidate metrics, noting filters and sample definitions differ.

## Interpretation and limits

- Ego-motion compensation is now applied to frame-to-frame box centers; the yaw audit is derived, not a source-data conversion.
- Motion-selected front/back assumes the target vehicle is driving forward. Reversing, lateral parking maneuvers, turns, or ID switches can produce a wrong sign; use the `heading_confidence` and `heading_source` columns and retain `heading_invalid` when evidence is weak.
- Propagation carries the sign branch, not a fixed absolute heading. Gaps over 0.75 s split a track segment.
- GPS position is used as the ego translation origin without applying a verified GNSS-antenna-to-LIDAR/ego lever arm. This can bias box-center velocity during ego turns; turn-rate filtering reduces but does not eliminate that risk. World yaw uses NAV orientation and is not affected by that translation offset.
- `Heading` and `Heading2` are both tested; empirical course agreement is not a substitute for the sensor vendor's field definition. NAV quality fields are preserved for manual filtering.
- Static objects without a moving anchor retain a valid box axis modulo 180 degrees where available, but have `heading_valid=false`; camera evidence is needed for front/back.
- Point-level motion compensation is separate. LAZ includes `point_timestamp`, but `LIDAR_MERGED_EGO` may already be deskewed. Do not apply another deskew without confirming the export convention and sensor-to-ego extrinsic.
- Candidate ranking is a diagnostic, not an automatic production label rewrite. Verify the winner on held-out tracks and camera overlays before adopting it.

## Artifacts

- `world_yaw_candidate_comparison.csv`
- `resolved_world_headings.csv`
- `nav_pose_coverage.csv`, `nav_heading_course_check.csv`, `nav_lidar_sync.csv`
- `world_yaw_candidate_motion_error.png`
