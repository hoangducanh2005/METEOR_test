# Task 2 — metrics & checklog

Rút gọn từ report của từng pipeline. Metrics so với Task 1 pseudo-GT; đây chưa phải accuracy trên human GT. `—` nghĩa là metric không được report cho pipeline đó.

## Batch 1 — 20260301

Dataset: `D:\mentor_data\PNKData\20260301_1443_VF6_01_1772350996_1772351781`  
71 frames × 6 camera = 426 ảnh.

| Pipeline / architecture | Drivable IoU | Lane F1 | Empty rate | BEV Lane F1 | BEV Chamfer (m) |
|---|---:|---:|---:|---:|---:|
| Mask2Former-Small + UFLD-v2 R18 (baseline) | 0.7690 | 0.0997 | 78.17% | — | 15.216 |
| Mask2Former-Small + UFLD-v2 R18 | — | 0.0985 | 78.40% | 0.0744 | 15.914 |
| Mask2Former-Small + CLRNet | — | 0.0876 | 86.15% | 0.0688 | 18.571 |
| Mask2Former Swin-L Mapillary (unified) | 0.8477 | 0.8710 | 8.69% | 0.3685 | 7.578 |

## Batch 2 — 20260228

Dataset: `D:\mentor_data\PNKData\20260228_1838_VF6_01_1772278707_1772279789`  
36 frames × 6 camera = 216 ảnh.

| Pipeline / architecture | Drivable IoU | Lane F1 | Empty rate | BEV Lane F1 | BEV Chamfer (m) |
|---|---:|---:|---:|---:|---:|
| Mask2Former-Small + UFLD-v2 R18 (baseline) | 0.6883 | 0.0252 | 91.20% | — | 10.679 |
| Mask2Former-Small + UFLD-v2 R18 | — | 0.0214 | 92.13% | 0.0068 | 12.064 |
| Mask2Former-Small + CLRNet | — | 0.0361 | 92.59% | 0.0189 | 15.837 |
| Mask2Former-Small + UFLD-v2 R34 | — | 0.0336 | 79.63% | 0.0110 | 15.298 |
| Mask2Former Swin-L Mapillary (unified) | 0.8719 | 0.3631 | 29.63% | 0.2090 | 3.366 |

## Checklog

| Batch | Images | Pipeline videos | Rendered frames | Validation |
|---|---:|---:|---:|---|
| Batch 1 | 426/426 | 4 | 71 per pipeline | `PASS_PROXY_ONLY`, failures=0 |
| Batch 2 | 216/216 | 5 | 36 per pipeline | `PASS_PROXY_ONLY`, failures=0 |

Mask2Former unified ran without UFLD-v2: all camera artifact sets and fused BEV frames were produced. Its image IoU/F1 reference is source-coupled to the same upstream Mask2Former checkpoint; treat as consistency evidence, not independent accuracy. BEV metrics use Task 1 pseudo-labels.

Detailed reports: `docs/Drivable_lane_detection/REPORT_RESULT.md` and each run's `out/task2_lane_bev/<scene>/` reports.
