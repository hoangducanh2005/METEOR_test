# Grounded SAM 2 Stop-Line Inference Report — Batch 3

## Kết quả

Đã chạy **Grounding DINO → SAM 2.1** trên toàn bộ 86 synchronized keyframe, sáu camera (516 ảnh), đồng thời chạy Mask2Former Mapillary trên cùng ảnh để xuất lane/drivable masks. Lệnh kết thúc thành công.

**Kết luận chất lượng:** chưa dùng output stop-line như annotation đáng tin cậy. Grounding DINO đưa ra box ở tất cả ảnh, nhưng box trung bình chiếm 29.2% diện tích ảnh; SAM 2.1 mask trung bình chiếm 29.6%, và 87.8% masks chiếm hơn 20% ảnh. Quan sát overlay `00000_CAM_FRONT_WIDE.jpg` cho thấy mask phủ phần lớn mặt đường thay vì đường stop line mảnh. Đây là coverage proxy và đánh giá định tính, không phải precision/recall vì không có stop-line pixel GT.

## Input và phạm vi

| Trường | Giá trị |
|---|---:|
| Raw folder | `D:\mentor_data\PNKData\20260305_1650_VF6_01_1772704234_1772704508` |
| Synchronized LiDAR keyframes | 86 |
| Camera images inferred | 516 (86 × 6) |
| Camera resolution | 1920 × 1536 |
| Cameras | FRONT_WIDE, FRONT_LEFT, FRONT_RIGHT, BACK_WIDE, BACK_LEFT, BACK_RIGHT |
| Camera sync delta p95 / max | 2.025 ms / 31.961 ms |
| Preflight sync limit | 35 ms |
| Device | CUDA — NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB |

Frame selection/synchronization được lấy từ `out/slsdnet/batch_3_preflight.json`. Inference dùng các camera image path trong manifest, không tải hoặc copy dataset.

## Model và cấu hình

- **Stop line:** [Grounding DINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) + [SAM 2.1 Hiera Small](https://huggingface.co/facebook/sam2.1-hiera-small). DINO prompt `stop line.`, box threshold `0.22`, text threshold `0.18`. Một SAM 2 video session được giữ cho mỗi camera; DINO cấp box mỗi frame để SAM 2 refine/refresh mask trong session.
- **Lane/drivable:** Mask2Former Swin-L Mapillary Vistas Panoptic, checkpoint `facebook/mask2former-swin-large-mapillary-vistas-panoptic`, taxonomy từ `pnk_lane_bev/configs/segmentation/mapillary_swin_large.yaml`.
- Implementation: `pnk_lane_bev/src/pnk_lane_bev/grounded_sam2_stopline.py`.

## Global metrics

| Metric | Giá trị | Cách đọc |
|---|---:|---|
| Grounded frames | 516 / 516 (100%) | Model phát box trong mọi ảnh; đây không phải detection recall |
| Grounding boxes | 1,295; 2.510 box/ảnh | Trung bình số box trước hợp nhất box thành một SAM prompt |
| Mean Grounding DINO score | 0.362 | Confidence model, không phải accuracy |
| Mean box area | 29.16% ảnh | Box vùng rộng, cần cảnh giác với prompt ambiguity |
| Mean SAM 2 stop mask area | 29.57% ảnh | Quá rộng cho stop-line marking dạng vạch mảnh |
| Masks có coverage >20% ảnh | 87.79% | Proxy over-segmentation; qualitative review cũng thấy mask phủ mặt đường |
| SAM-only propagated nonempty masks | 0 frame | DINO có prompt ở từng frame nên run không kiểm tra khả năng giữ mask khi thiếu detection |
| Mean Mask2Former lane-mask coverage | 3.91% ảnh | Coverage, không phải lane accuracy |
| Mean Mask2Former drivable-mask coverage | 29.50% ảnh | Coverage, không phải drivable IoU |
| Mean end-to-end latency | 3,384 ms / camera image | DINO + SAM 2 + Mask2Former, tính trong loop |
| Wall time | 1,778.98 s (29m 39s) | 516 ảnh trên GPU nêu trên |
| Stop-line precision / recall / F1 / IoU | N/A | Không có ground-truth masks |
| BEV stop-line / IPM metrics | N/A | Xem phần calibration bên dưới |

## Per-camera breakdown

Coverage là tỷ lệ số pixel mask trên diện tích ảnh. Latency là trung bình của pipeline DINO + SAM 2 + Mask2Former theo camera image.

| Camera | Grounded frames | Boxes/frame | Mean DINO score | Stop mask coverage | Lane coverage | Drivable coverage | Latency ms/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| FRONT_WIDE | 86/86 (100%) | 2.314 | 0.386 | 30.73% | 4.16% | 31.66% | 1,907 |
| FRONT_LEFT | 86/86 (100%) | 2.767 | 0.333 | 19.02% | 2.81% | 18.67% | 6,623 |
| FRONT_RIGHT | 86/86 (100%) | 2.407 | 0.361 | 29.44% | 2.97% | 27.41% | 2,118 |
| BACK_WIDE | 86/86 (100%) | 2.988 | 0.325 | 26.27% | 3.16% | 24.61% | 3,151 |
| BACK_LEFT | 86/86 (100%) | 2.198 | 0.399 | 29.84% | 5.21% | 32.14% | 1,358 |
| BACK_RIGHT | 86/86 (100%) | 2.384 | 0.386 | 42.15% | 5.17% | 42.50% | 5,150 |

BACK_RIGHT và FRONT_LEFT có latency cao trong lượt chạy này; cần đo lại nếu so sánh tốc độ model nghiêm túc. Stop-mask coverage cao ở tất cả camera, đặc biệt BACK_RIGHT.

## Checklog

```text
==============================================================================
MODEL LABELS USED
==============================================================================
  lane marking : ['Lane Marking - General', 'Lane Marking - Crosswalk']
  road surface : ['Road', 'Service Lane', 'Bike Lane', 'Parking',
                  'Crosswalk - Plain', 'Lane Marking - General',
                  'Lane Marking - Crosswalk']
  stop line    : Grounding DINO text prompt ['stop line.'] -> SAM 2.1 mask

==============================================================================
INPUT / SYNCHRONIZATION
==============================================================================
  raw folder   : D:\mentor_data\PNKData\20260305_1650_VF6_01_1772704234_1772704508
  keyframes    : 86 / accepted 86 / dropped 0
  camera views : FRONT_WIDE, FRONT_LEFT, FRONT_RIGHT,
                 BACK_WIDE, BACK_LEFT, BACK_RIGHT
  images       : 516 at 1920 x 1536
  sync delta   : p95=2.025 ms, max=31.961 ms, limit=35 ms

==============================================================================
INFERENCE / COVERAGE PROXIES (NOT ACCURACY METRICS)
==============================================================================
  Grounding DINO boxes       : 1295 total; 2.510 / image
  Grounded image coverage    : 516 / 516 (100%)
  Mean DINO box score        : 0.362
  Mean box area              : 29.16% of image
  Mean SAM 2 stop mask       : 29.57% of image
  Stop masks >20% image      : 87.79%
  SAM-only propagation masks : 0 frames
  Mean Mask2Former lane      : 3.91% of image
  Mean Mask2Former drivable  : 29.50% of image
  Mean latency               : 3384 ms / image
  Wall time                  : 1778.98 s; CUDA RTX 4050 Laptop
  Human stop-line GT         : unavailable

==============================================================================
CALIBRATION / BEV
==============================================================================
  Intrinsic/extrinsic calibration for this raw folder: not found in CAMERA,
  LIDAR, or OTHERS. No IPM or BEV metric was generated.
```

## Deliverables

- Full metrics and per-image boxes/scores/coverage/latency: `out/grounded_sam2/batch_3/metrics.json`.
- Stop-line masks: `out/grounded_sam2/batch_3/masks/stop_line/`.
- Mask2Former semantic, lane, drivable masks: `out/grounded_sam2/batch_3/masks/{semantic,lane,drivable}/`.
- RGB overlays: `out/grounded_sam2/batch_3/overlays/` (SAM stop region in red, DINO boxes in yellow, drivable in green, lane in yellow/cyan).
- One-frame smoke artifacts: `out/grounded_sam2/batch_3_smoke/`.

## Calibration and accuracy limitations

The raw folder has no calibration files in `CAMERA`, `LIDAR`, or `OTHERS`. METEOR calibration exists for a different VF6 recording, but this report does not assume it matches the current recording. Therefore no image-to-BEV/IPM projection is reported. There is also no human stop-line GT, so high grounding coverage must not be interpreted as high precision. The full-run overlays indicate that the chosen prompt/checkpoints over-segment road regions; this branch needs prompt/geometry refinement and human review before its masks are used as stop-line labels.

## Reproduce

```powershell
$env:HF_HOME = Join-Path (Get-Location) 'out/grounded_sam2/hf_cache'
$env:HF_HUB_DISABLE_XET = '1'
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'pnk_lane_bev/src'
& '.\pnk_comet_autolabel\.venv\Scripts\python.exe' `
  'pnk_lane_bev\src\pnk_lane_bev\grounded_sam2_stopline.py' `
  --preflight 'out\slsdnet\batch_3_preflight.json' `
  --output 'out\grounded_sam2\batch_3' --device cuda
```
