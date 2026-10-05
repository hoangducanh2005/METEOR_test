# Task 2 — Mask2Former Mapillary on RLMD Validation

## Kết quả

Đã chạy trực tiếp Mask2Former Swin-L Mapillary Vistas trên toàn bộ **400 ảnh validation** RLMD (1920 × 1080). Lane-marking và crosswalk có GT image-space; bộ này **không có road/drivable-area GT đầy đủ**, nên không thể tính Drivable IoU hợp lệ.

| Nhóm đánh giá | IoU | F1 | Precision | Recall | Prediction empty |
|---|---:|---:|---:|---:|---:|
| Lane marking (GT IDs 3–10, 16) | **26.10%** | **41.39%** | 32.84% | 55.97% | 3.75% |
| Stop-line subset (GT ID 3) | **1.43%** | **2.81%** | 1.45% | 49.88% | 3.75% |
| Crosswalk (GT ID 2) | **40.68%** | **57.84%** | 81.63% | 44.78% | 80.00% |
| Drivable area | N/A | N/A | N/A | N/A | N/A |

Mean lane IoU trên ảnh có GT lane marking là **31.22%**; crosswalk mean IoU trên ảnh GT-positive là **24.24%**. Lane prediction chiếm trung bình 1.51% diện tích ảnh, còn drivable prediction chiếm 11.44%; đây chỉ là coverage, không phải accuracy.

## Stop-line cần lưu ý

Mask2Former Mapillary **không có class stop-line riêng**. Stop-line chỉ có thể rơi vào `Lane Marking - General` hoặc bị bỏ sót; cùng mask cũng chứa lane dividers và các vạch khác. Khi so toàn bộ lane output với riêng GT stop-line, kết quả là IoU **1.43%**, precision **1.45%**, recall **49.88%**. Model phủ được một phần stop-line nhưng nhầm quá nhiều pixel khác, nên không thể dùng output này làm stop-line annotation chuyên biệt.

So với Task 4(a) Grounded SAM 2 trên đúng 400 ảnh và đúng GT, lane output của Mask2Former có stop-line IoU cao hơn (1.43% so với 0.13%), nhưng cả hai đều quá thấp để coi là stop-line detector đạt yêu cầu. Đây là phép so pixel-mask; Mask2Former metric dùng generic lane-marking mask, không phải class stop-line.

## Dataset và GT mapping

| Trường | Giá trị |
|---|---|
| Dataset | `C:\Users\Admin\Downloads\RLMD_1080p-20261001T052305Z-1-001\RLMD_1080p` |
| Split / số ảnh | `val` / 400 |
| Ảnh-mask khớp tên | 400 / 400 |
| Resolution | 1920 × 1080 |
| GT lane marking | IDs 3–10, 16: stop line, solid/dashed/double line, channelizing line |
| GT crosswalk | ID 2: `crosswalk` |
| GT drivable | Không có class `road`/drivable; background không được xem là non-drivable |

Lane metric đối chiếu Mapillary `Lane Marking - General` + `Lane Marking - Crosswalk` với GT line classes 3–10 và 16. Vì RLMD mã hóa crosswalk bằng class riêng ID 2, crosswalk được tính thêm thành metric riêng với Mapillary `Crosswalk - Plain` + `Lane Marking - Crosswalk`. Stop-line subset xem toàn bộ lane output như dự đoán stop-line; các vạch lane không phải stop-line trở thành false positive trong metric này.

## Model và cấu hình

- Checkpoint: `facebook/mask2former-swin-large-mapillary-vistas-panoptic` (65-class Mapillary Vistas panoptic).
- Drivable prediction classes: Road, Service Lane, Bike Lane, Parking, Crosswalk - Plain, Lane Marking - General, Lane Marking - Crosswalk.
- Lane prediction classes: Lane Marking - General, Lane Marking - Crosswalk.
- Confidence threshold: `0.70` (profile FRONT_WIDE; RLMD không có camera ID để chọn ngưỡng theo camera).
- Lane post-processing: close kernel 3, remove components <12 px, không skeletonize.
- GPU: NVIDIA GeForce RTX 4050 Laptop, CUDA.
- Transformers loading report có hai tham số `swin.layernorm.weight/bias` thiếu và được khởi tạo mới; các `relative_position_index` là unexpected keys. Metrics phản ánh chính xác lần chạy checkpoint trong môi trường hiện tại, nhưng cảnh báo tương thích này cần được xử lý trước khi xem đây là benchmark cuối cùng.

## Checklog

```text
==============================================================================
MODEL / TAXONOMY
==============================================================================
  model          : facebook/mask2former-swin-large-mapillary-vistas-panoptic
  taxonomy       : Mapillary Vistas, 65 classes
  lane output    : Lane Marking - General, Lane Marking - Crosswalk
  drive output   : Road, Service Lane, Bike Lane, Parking, Crosswalk - Plain,
                   Lane Marking - General, Lane Marking - Crosswalk
  confidence     : 0.70 (FRONT_WIDE fallback; no camera ID in RLMD)
  lane postproc  : close=3, min_component=12 px, skeletonize=false
  stop-line class: unavailable; evaluated via generic lane output only

==============================================================================
INPUT / GROUND TRUTH
==============================================================================
  split          : val
  images         : 400; paired masks: 400
  resolution     : 1920 x 1080
  lane GT        : RLMD IDs 3-10,16; 397 positive images
  stop-line GT   : RLMD ID 3; 141 positive images
  crosswalk GT   : RLMD ID 2; 151 positive images
  drivable GT    : unavailable (no road/drivable-area class)

==============================================================================
IMAGE-SPACE METRICS (PIXEL MICRO)
==============================================================================
  lane IoU / F1 / P / R : 0.260968 / 0.413917 / 0.328376 / 0.559721
  lane empty rate       : 3.75%
  stop IoU / F1 / P / R : 0.014265 / 0.028129 / 0.014473 / 0.498836
  cross IoU/F1/P/R      : 0.406837 / 0.578371 / 0.816273 / 0.447847
  crosswalk empty rate  : 80.00%
  mean drive coverage   : 11.44% (not accuracy)
  mean lane coverage    : 1.51%
  drivable IoU          : N/A

==============================================================================
RUNTIME / GEOMETRY
==============================================================================
  model latency mean   : 379.2 ms/image
  total model latency  : 151.69 s
  wall time            : 887.05 s
  calibration          : unavailable
  BEV / IPM metrics    : N/A

==============================================================================
WEIGHT LOADING NOTE
==============================================================================
  MISSING: swin.layernorm.weight, swin.layernorm.bias (newly initialized)
  UNEXPECTED: relative_position_index keys (ignored by loader)
```

## Deliverables

- Metrics and per-image breakdown: [`metrics.json`](../../out/mask2former/task_2_rlmd_1080p/metrics.json)
- Semantic predictions (uint16 class IDs): `out/mask2former/task_2_rlmd_1080p/masks/semantic/`
- Lane masks: `out/mask2former/task_2_rlmd_1080p/masks/lane_marking/`
- Drivable masks: `out/mask2former/task_2_rlmd_1080p/masks/drivable/`
- Lane vs GT overlays (red = prediction only, green = GT only, yellow = overlap): `out/mask2former/task_2_rlmd_1080p/overlays/`
- Example: [`43690.jpg`](../../out/mask2former/task_2_rlmd_1080p/overlays/43690.jpg)
- Reproducible evaluator: [`evaluate_mask2former_mapillary_rlmd.py`](../../pnk_lane_bev/src/pnk_lane_bev/evaluate_mask2former_mapillary_rlmd.py)

## Conclusion

On this dataset Mask2Former Mapillary performs moderately on the broad lane-marking union (IoU 26.10%, F1 41.39%) and has a better but recall-limited crosswalk result (IoU 40.68%, recall 44.78%). Its generic lane-marking output does **not** isolate stop lines (IoU 1.43%, precision 1.45%). Drivable quality cannot be measured from RLMD labels, and no BEV metrics can be produced without camera calibration.
