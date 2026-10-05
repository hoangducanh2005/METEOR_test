# Task 4(a) — Grounded SAM 2 on RLMD_1080p Validation

## Kết quả

Đã chạy Grounding DINO Tiny + SAM 2.1 Hiera Small trên toàn bộ **400 ảnh validation** của RLMD. Bộ này có pixel label cho class `stop line`, nên có thể tính metrics image-space. Kết quả cho thấy nhánh hiện tại **không đủ chính xác để dùng làm stop-line annotation**: model phủ vùng lớn của mặt đường thay vì chỉ vạch stop-line.

| Metric | Kết quả |
|---|---:|
| Pixel micro IoU | **0.1275%** |
| Pixel micro F1 | **0.2547%** |
| Pixel precision | **0.1276%** |
| Pixel recall | **71.69%** |
| Mean IoU trên ảnh có GT stop-line | **0.5218%** |
| Ảnh GT có stop-line | 141 / 400 (35.25%) |
| GT empty rate | 64.75% |
| Prediction empty rate | **0.00%** |
| Miss rate trên ảnh GT-positive | 0.00% |
| GT stop-line pixels | 363,330 |
| Predicted stop-line pixels | 204,191,645 |
| TP / FP / FN pixels | 260,468 / 203,931,177 / 102,862 |
| Mean inference latency | 486.7 ms / ảnh |
| Tổng thời gian inference | 293.85 s (4m 54s) |

Recall cao không đồng nghĩa với mask tốt ở đây: model dự đoán không rỗng trên cả 400 ảnh, kể cả ảnh GT-rỗng. Tổng vùng dự đoán lớn khoảng **562 lần** tổng vùng GT; vì vậy precision và IoU gần bằng 0. Điều này khớp với overlay quan sát được: mask đỏ phủ phần lớn mặt đường.

## Dataset và nhãn

| Trường | Giá trị |
|---|---|
| Dataset folder | `C:\Users\Admin\Downloads\RLMD_1080p-20261001T052305Z-1-001\RLMD_1080p` |
| Split được đánh giá | `val` |
| Số ảnh validation | 400 |
| Kích thước ảnh/mask | 1920 × 1080 |
| Số cặp ảnh-mask khớp tên | 400 / 400 |
| Class GT | ID `3`, `stop line` |
| Màu RGB lấy từ `rlmd.csv` | `(61, 72, 204)` |

Mask được giải mã bằng đối chiếu chính xác màu RGB trong `rlmd.csv`; ảnh và nhãn dùng cùng stem tên file. Ví dụ GT-only và prediction-vs-GT overlays đã được lưu để kiểm tra trực quan.

## Model và cấu hình

- **Grounding DINO Tiny** tìm vùng bằng prompt `stop line.`.
- **SAM 2.1 Hiera Small** segment vùng box hợp nhất từ các detection.
- Giữ ngưỡng của Task 3: box threshold `0.22`, text threshold `0.18`.
- SAM mask threshold `0.0`, morphology close kernel `5×3`, bỏ component nhỏ hơn 20 pixel.
- Dùng cùng checkpoint SAM 2.1 Hiera Small đã dùng ở Task 3; validation được xử lý độc lập từng ảnh vì bộ RLMD không cung cấp thứ tự video/sync metadata đáng tin cậy để đánh giá temporal propagation. Do đó metrics ở đây đo Grounding DINO + SAM image segmentation, không đo tracking.
- GPU: NVIDIA GeForce RTX 4050 Laptop, CUDA.

## Checklog

```text
==============================================================================
DATASET / GROUND TRUTH
==============================================================================
  dataset       : RLMD_1080p
  split         : val
  images        : 400 / 400 paired with labels
  resolution    : 1920 x 1080
  class         : id=3, stop line
  label color   : RGB (61, 72, 204), read from rlmd.csv
  GT-positive   : 141 / 400 (35.25%)
  GT empty rate : 64.75%

==============================================================================
MODEL
==============================================================================
  detector      : IDEA-Research/grounding-dino-tiny
  segmenter     : facebook/sam2.1-hiera-small
  prompt        : "stop line."
  box threshold : 0.22
  text threshold: 0.18
  inference     : independent image masks; temporal propagation disabled

==============================================================================
IMAGE-SPACE METRICS
==============================================================================
  micro IoU            : 0.001275 (0.1275%)
  micro F1             : 0.002547 (0.2547%)
  micro precision      : 0.001276 (0.1276%)
  micro recall         : 0.716891 (71.69%)
  mean IoU, GT-positive: 0.005218 (0.5218%)
  prediction empty rate: 0.00%
  GT empty rate        : 64.75%
  miss rate, GT-positive: 0.00%
  TP / FP / FN pixels  : 260,468 / 203,931,177 / 102,862
  predicted / GT pixels: 204,191,645 / 363,330

==============================================================================
RUNTIME / BEV
==============================================================================
  mean latency         : 486.7 ms/image
  total inference      : 293.85 s
  intrinsic/extrinsic  : unavailable
  BEV IoU / IPM metrics: N/A
```

## Deliverables

- Full metrics and per-image results: [`metrics.json`](../../out/grounded_sam2/task_4a_rlmd_1080p/metrics.json)
- 400 grayscale predicted masks: `out/grounded_sam2/task_4a_rlmd_1080p/masks/pred_stopline/`
- 400 overlays: `out/grounded_sam2/task_4a_rlmd_1080p/overlays/`
- Overlay colors: red = prediction only, green = GT only, yellow = overlap.
- Example overlay: [`43690.jpg`](../../out/grounded_sam2/task_4a_rlmd_1080p/overlays/43690.jpg)
- Example GT-only visualization: [`gt_only_43690.jpg`](../../out/grounded_sam2/task_4a_rlmd_1080p/gt_only_43690.jpg)
- Reproducible evaluator: [`evaluate_grounded_sam2_rlmd.py`](../../pnk_lane_bev/src/pnk_lane_bev/evaluate_grounded_sam2_rlmd.py)

## Conclusion and limits

Grounded SAM 2 finds stop-line-like regions often enough to achieve 71.7% pixel recall, but its masks are far too broad: precision is 0.13%, micro IoU is 0.13%, and prediction empty rate is 0%. It therefore fails this validation benchmark and should not generate stop-line labels for training without a more constrained detector/refinement stage.

No intrinsic or extrinsic calibration was provided with this RLMD folder, so BEV projection and BEV IoU were not computed. The report evaluates only the stop-line class; it does not evaluate crosswalk or lane marking classes.
