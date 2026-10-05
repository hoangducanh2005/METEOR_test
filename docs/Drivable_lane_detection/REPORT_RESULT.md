# Task 2 — Architecture Metrics Report



## 4. Global metrics

| Pipeline | Lane F1 ↑ | Empty rate ↓ | Lane/frame ↑ | Mean confidence ↑ | Runtime/image ↓ | BEV Lane F1 ↑ | BEV Chamfer ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy UFLD + max fusion | **0.0997** | **78.17%** | 0.523 | N/A | **27.46 ms** | N/A | **15.216 m** |
| UFLD-v2 + new preprocessing/fusion | 0.0985 | 78.40% | **0.528** | **0.1864** | 42.29 ms | **0.0744** | 15.914 m |
| CLRNet + new preprocessing/fusion | 0.0876 | 86.15% | 0.364 | 0.0863 | 35.55 ms | 0.0688 | 18.571 m |

### 4.1 CLRNet so với UFLD-v2 cùng pipeline mới

| Metric | CLRNet delta | Đánh giá |
|---|---:|---|
| Lane F1 | `-0.0109` | Giảm khoảng 11.1% tương đối |
| Empty rate | `+7.75` điểm % | CLRNet trả prediction rỗng thường xuyên hơn |
| Lane/frame | `-0.164` | Giảm khoảng 31.1% |
| Mean confidence | `-0.1001` | Confidence trung bình thấp hơn đáng kể |
| Runtime/image | `-6.74 ms` | CLRNet nhanh hơn UFLD-v2 mới khoảng 15.9% |
| BEV Lane F1 | `-0.0056` | Chất lượng BEV giảm |
| BEV Chamfer | `+2.657 m` | Sai lệch hình học BEV tăng khoảng 16.7% |

CLRNet có lợi thế runtime trong lần chạy này, nhưng không bù được mức giảm detection coverage và BEV quality.

## 5. Breakdown theo camera

### 5.1 Lane F1

| Camera | Legacy UFLD | UFLD-v2 mới | CLRNet | CLRNet so với UFLD-v2 |
|---|---:|---:|---:|---:|
| FRONT_WIDE | 0.1312 | **0.1360** | 0.1274 | -0.0086 |
| FRONT_LEFT | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FRONT_RIGHT | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| BACK_WIDE | **0.4482** | 0.4389 | 0.3982 | -0.0408 |
| BACK_LEFT | **0.0185** | 0.0161 | 0.0000 | -0.0161 |
| BACK_RIGHT | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### 5.2 Empty prediction rate

| Camera | Legacy UFLD | UFLD-v2 mới | CLRNet |
|---|---:|---:|---:|
| FRONT_WIDE | 69.01% | 69.01% | 76.06% |
| FRONT_LEFT | 100.00% | 100.00% | 100.00% |
| FRONT_RIGHT | 98.59% | 98.59% | 100.00% |
| BACK_WIDE | **15.49%** | 16.90% | 40.85% |
| BACK_LEFT | 85.92% | 85.92% | 100.00% |
| BACK_RIGHT | 100.00% | 100.00% | 100.00% |

### 5.3 Lane/frame, confidence, runtime và per-camera BEV Chamfer

| Camera | Detector | Lane/frame | Mean conf. | Runtime | BEV Chamfer |
|---|---|---:|---:|---:|---:|
| FRONT_WIDE | UFLD-v2 | 0.479 | 0.255 | 61.08 ms | 6.863 m |
| FRONT_WIDE | CLRNet | 0.408 | 0.160 | 40.53 ms | **4.816 m** |
| FRONT_LEFT | UFLD-v2 | 0.000 | 0.000 | 42.88 ms | N/A — empty |
| FRONT_LEFT | CLRNet | 0.000 | 0.000 | 35.75 ms | N/A — empty |
| FRONT_RIGHT | UFLD-v2 | 0.014 | 0.011 | 38.80 ms | 9.969 m |
| FRONT_RIGHT | CLRNet | 0.000 | 0.000 | 34.30 ms | N/A — empty |
| BACK_WIDE | UFLD-v2 | **2.479** | **0.738** | 33.90 ms | **12.498 m** |
| BACK_WIDE | CLRNet | 1.775 | 0.359 | 33.91 ms | 14.395 m |
| BACK_LEFT | UFLD-v2 | 0.197 | 0.114 | 37.03 ms | 1.453 m |
| BACK_LEFT | CLRNet | 0.000 | 0.000 | 34.67 ms | N/A — empty |
| BACK_RIGHT | UFLD-v2 | 0.000 | 0.000 | 40.03 ms | N/A — empty |
| BACK_RIGHT | CLRNet | 0.000 | 0.000 | 34.13 ms | N/A — empty |

CLRNet có per-camera Chamfer tốt hơn trên `FRONT_WIDE` khi có prediction, nhưng coverage thấp hơn. Fused BEV cuối cùng vẫn kém hơn vì prediction rỗng nhiều và `BACK_WIDE` suy giảm mạnh.

## 6. Checklog

Checklog dùng format cố định dưới đây để review model labels, LiDAR intensity, calibration cross-check và lane-marking mask theo khoảng cách.

```text
==============================================================================
MODEL LABELS USED
==============================================================================
  lane marking : ['Lane Marking - General']
  road surface : ['Lane Marking - General', 'Road']
 
==============================================================================
INTENSITY on model-ROAD ground points (pick --paint-min from this if needed)
==============================================================================
  range    asphalt p50    p90    p99  p99.9    max    paint n paint p10    p50
  0-5m             9.0   10.0   15.0   22.0     35        531       9.0   10.0
  5-10m            6.0    9.0   12.0   23.0     42       3688      11.0   18.0
  10-15m           2.0    3.0    4.0    7.0      7        983       9.0    9.0
  15-20m           2.0    3.0    4.0    7.0      9       1881       9.0   17.0
  20-25m           2.0    2.0    4.0    7.0      9       1733       9.0   17.0
  25-30m           2.0    2.0    4.0    9.0      9        920       9.0   23.0
  30-35m           1.0    2.0    3.0    4.0      9        616       7.0   17.0
  35-40m           1.0    2.0    3.0    4.0      7        304       7.0    7.0
  40-45m           1.0    2.0    3.0    4.0      4        101       7.0    9.0
  45-50m           1.0    2.0    3.0    4.0      4        227       7.0    9.0
  50-55m           1.0    2.0    4.0    4.0      4        277       7.0    9.0
  55-60m           1.0    3.0    4.0    9.0      9        143       7.0    9.0
  Paint should be CLEARLY separated from asphalt (paint p10 >> asphalt p99). If not, set --paint-min.
 
==============================================================================
(A) CALIBRATION CROSS-CHECK on lane paint (<30 m)
==============================================================================
  CAM_P_F   score@(0,0)=0.161  best-shift centre=(+1.0, -5.3)  gain=+0.013
  CAM_P_FL  score@(0,0)=0.221  best-shift centre=(-2.8, +3.5)  gain=+0.003
  CAM_P_FR  score@(0,0)=0.014  best-shift centre=(+0.6, -3.6)  gain=+0.000
  Only meaningful when score@(0,0) is clearly > 0.1; tiny scores = no signal, ignore the shift.
 
==============================================================================
(B) LANE-MARKING MASK QUALITY on model road (tolerance 3px)
==============================================================================
  cam       range m  road-cover   paint  recall@0 recall@tol false-mark   lift
  CAM_P_F   0-10          0.819     752     0.331      0.420      0.018   16.9
  CAM_P_F   10-20         0.920     867     0.292      0.323      0.026   10.8
  CAM_P_F   20-30         0.879    1308     0.013      0.013      0.003    4.3
  CAM_P_F   30-45         0.683     332     0.000      0.000      0.000    0.0
  CAM_P_F   45-70         0.147      11   (too little paint)
  CAM_P_FL  0-10          0.578    3340     0.248      0.257      0.091    2.6
  CAM_P_FL  10-20         0.263     921     0.542      0.564      0.132    4.0
  CAM_P_FL  20-30         0.256    1020     0.352      0.385      0.069    4.9
  CAM_P_FL  30-45         0.169     473     0.004      0.015      0.021    0.2
  CAM_P_FL  45-70         0.079       3   (too little paint)
  CAM_P_FR  0-10          0.804     127     0.142      0.142      0.002   91.0
  CAM_P_FR  10-20         0.921    1076     0.005      0.005      0.001    4.1
  CAM_P_FR  20-30         0.769     325     0.000      0.000      0.000    0.0
  CAM_P_FR  30-45         0.615     216     0.000      0.000      0.000    0.0
  CAM_P_FR  45-70         0.520     638     0.000      0.000      0.000    0.0
```

## 7. Đánh giá vùng quan tâm 40–70 m

### 7.1 Intensity separation ở 40–60 m

| Range | Asphalt p99 | Asphalt p99.9 | Paint n | Paint p10 | Paint p50 | Kết luận |
|---|---:|---:|---:|---:|---:|---|
| 40–45 m | 3 | 4 | 101 | 7 | 9 | Tách rõ ở percentile chính |
| 45–50 m | 3 | 4 | 227 | 7 | 9 | Tách rõ |
| 50–55 m | 4 | 4 | 277 | 7 | 9 | Tách được, margin nhỏ hơn |
| 55–60 m | 4 | 9 | 143 | 7 | 9 | Paint p10 vượt asphalt p99, nhưng asphalt tail chồng lấn |
| 60–70 m | N/A | N/A | N/A | N/A | N/A | Checklog chưa cung cấp intensity bins |

Trong 40–60 m, paint p10 luôn bằng `7`, còn asphalt p99 nằm trong `3–4`. Intensity vẫn có tín hiệu hữu ích cho far-range paint validation. Một giá trị thử nghiệm ban đầu hợp lý là:

```text
--paint-min 6
```

Giá trị này nằm giữa asphalt p99 lớn nhất (`4`) và paint p10 nhỏ nhất (`7`). Cần sweep `{5, 6, 7}` trên held-out frames vì bin 55–60 m có asphalt p99.9 bằng `9`, cho thấy tail overlap. Checklog chưa đủ dữ liệu để khóa threshold cho 60–70 m.

### 7.2 Lane-marking mask ở 40–70 m

Evaluator hiện dùng bin `30–45 m` và `45–70 m`; vì vậy không thể tách riêng `40–45 m` từ số liệu được cung cấp. Bin `30–45 m` chỉ được dùng làm chỉ báo gần nhất cho đoạn đầu vùng quan tâm.

| Camera | 30–45 m proxy | 45–70 m | Đánh giá far-range |
|---|---|---|---|
| `CAM_P_F` | 332 paint, recall@tol `0.000`, lift `0.0` | 11 paint, quá ít để chấm | Mask mất lane trước hoặc khi đến 40–45 m; chưa đủ sample sau 45 m |
| `CAM_P_FL` | 473 paint, recall@tol `0.015`, false-mark `0.021`, lift `0.2` | 3 paint, quá ít để chấm | Recall gần như mất hoàn toàn; 45–70 m không đủ sample |
| `CAM_P_FR` | 216 paint, recall@tol `0.000`, lift `0.0` | 638 paint, recall@tol `0.000`, lift `0.0` | Có đủ paint evidence ở 45–70 m nhưng model không bắt được lane marking |

`CAM_P_FR` là bằng chứng mạnh nhất cho lỗi far range: road cover vẫn `0.520`, có `638` paint points, nhưng recall bằng 0 ngay cả với tolerance 3 px. Đây là failure của lane-marking mask/detector trong vùng xa theo proxy hiện tại.

`CAM_P_F` và `CAM_P_FL` không có đủ paint points trong 45–70 m để kết luận accuracy. Việc gán score 0 cho hai camera này sẽ trộn lẫn model failure với thiếu reference support; report giữ trạng thái `too little paint`.

### 7.3 Calibration cross-check

| Camera | score@(0,0) | Best shift | Gain | Đánh giá |
|---|---:|---:|---:|---|
| `CAM_P_F` | 0.161 | (+1.0, -5.3) px | +0.013 | Có signal; shift dọc đáng kiểm tra nhưng gain nhỏ |
| `CAM_P_FL` | 0.221 | (-2.8, +3.5) px | +0.003 | Có signal; correction gần như không cải thiện score |
| `CAM_P_FR` | 0.014 | (+0.6, -3.6) px | +0.000 | Không có signal đủ mạnh; bỏ qua shift |

Calibration correction nhỏ không giải thích được mức recall 0 trong toàn bộ vùng 30–70 m. Với `CAM_P_F` và `CAM_P_FL`, best-shift gain chỉ `0.013` và `0.003`. Với `CAM_P_FR`, score gốc quá thấp để shift có ý nghĩa. Detector/mask quality vẫn là bottleneck cần xử lý trước.

### 7.4 Rủi ro proxy coupling

Checklog dùng:

```text
lane marking : Lane Marking - General
road surface : Lane Marking - General + Road
```

Lane class đồng thời nằm trong road-surface gating. Điều này hợp lý để không loại lane pixels khỏi road, nhưng metric không hoàn toàn độc lập giữa model lane mask và model-road support. Human-reviewed paint/lane annotations vẫn cần cho kết luận accuracy.




## 12. Architecture 3 — Unified Mask2Former, không dùng UFLD-v2


Nguồn model và architecture:

- [Mask2Former Swin-L Mapillary Vistas Panoptic](https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-panoptic)
- [Mask2Former source repository](https://github.com/facebookresearch/Mask2Former)


### 12.2 Class được sử dụng

```text
drivable:
  Road
  Service Lane
  Bike Lane
  Parking
  Crosswalk - Plain
  Lane Marking - General
  Lane Marking - Crosswalk

lane:
  Lane Marking - General
  Lane Marking - Crosswalk
```

Model taxonomy được resolve theo `id2label`: `Road=13`, `Service Lane=14`, `Bike Lane=7`, `Lane Marking - Crosswalk=23`, `Lane Marking - General=24`. Pipeline fail fast nếu checkpoint thay đổi và thiếu class cấu hình.

### 12.4 So sánh trực tiếp với baseline cũ

| Metric | Mask2Former-Small + UFLD-v2 | Unified Mask2Former Swin-L | Delta |
|---|---:|---:|---:|
| Image Drivable IoU ↑ | 0.7690 | **0.8477** | +0.0787 |
| Image Lane F1 ↑ | 0.0997 | **0.8710** | +0.7713 |
| Empty lane rate ↓ | 78.17% | **8.69%** | -69.48 điểm % |
| BEV Drivable IoU ↑ | 0.2382 | **0.2417** | +0.0035 |
| BEV Lane F1 ↑ | N/A | 0.3685 | N/A |
| BEV Lane Chamfer ↓ | 15.216 m | **7.578 m** | -7.638 m |

Lane post-processing thay đổi proxy F1 từ `0.8721` raw xuống `0.8710` sau morphology/component filtering, delta `-0.0011`. Mức giảm rất nhỏ; hậu xử lý đổi lại mask ít component nhiễu hơn. Skeletonization đã implement nhưng tắt trong config baseline này.

### 12.5 Breakdown theo camera

| Camera | Drive IoU | Lane F1 | Empty rate | Components/frame | Mean conf. | BEV Drive IoU | BEV Chamfer |
|---|---:|---:|---:|---:|---:|---:|---:|
| FRONT_WIDE | 0.8681 | 0.9058 | 0.00% | 6.437 | 0.999 | 0.2372 | 3.277 m |
| FRONT_LEFT | 0.8660 | 0.6610 | **33.80%** | 2.507 | 0.658 | 0.3316 | 2.202 m |
| FRONT_RIGHT | 0.7723 | **0.9464** | 1.41% | 1.493 | 0.982 | 0.3263 | **9.084 m** |
| BACK_WIDE | 0.8578 | 0.9064 | 2.82% | 9.972 | 0.970 | 0.2256 | 7.161 m |
| BACK_LEFT | **0.8726** | 0.9146 | 8.45% | 3.972 | 0.905 | **0.4959** | **1.692 m** |
| BACK_RIGHT | 0.8493 | 0.8918 | 5.63% | 1.915 | 0.917 | 0.4071 | 8.697 m |

`FRONT_LEFT` vẫn yếu nhất về coverage với empty rate `33.80%`. `FRONT_RIGHT` và `BACK_RIGHT` có image-space proxy F1 cao nhưng BEV Chamfer lần lượt `9.084 m` và `8.697 m`; đây là dấu hiệu camera-specific IPM/calibration cần audit.

### 12.6 Runtime

| Stage | Runtime |
|---|---:|
| Mask2Former direct inference | 83.26 ms/image, mean trên 12 timed samples |
| Lane/drivable class filter + post-process | 8.57 ms/image, mean trên 426 ảnh |
| IPM + six-camera fusion | 418.10 ms/frame, mean trên 71 frame |
| Baseline Small + UFLD model branches | 286.77 ms/image |

Direct runtime dùng RTX 4050 Laptop GPU, FP16, một warmup và hai timed frame cho mỗi camera. Đây là sampled benchmark; không so sánh tuyệt đối với full-run latency cũ vì protocol và phiên bản runtime khác nhau. Transformers `5.17.0` báo hai Swin LayerNorm parameters được khởi tạo mặc định khi load snapshot cũ; quality metrics full folder dùng panoptic cache đã sinh trước đó từ cùng checkpoint, còn direct load được dùng để kiểm tra code path và đo runtime.



## 13. Dataset run 2 — `pnk_132_1772278707_1772278713`

### 13.1 Input và phạm vi chạy

Pipeline Unified Mask2Former ở mục 12 được giữ nguyên; lần chạy này chỉ đổi raw dataset:

```text
D:/mentor_data/PNKData/20260228_1838_VF6_01_1772278707_1772279789
  → COMET replacement / general-purpose annotation
  → METEOR Autolabel Factory
  → Mask2Former unified drivable + lane
  → calibrated IPM + confidence-weighted six-camera fusion
  → metrics + video 6 camera/BEV
```

| Preflight check | Kết quả |
|---|---:|
| Ảnh có sẵn trên mỗi raw camera | 211 |
| LiDAR keyframes dùng để đồng bộ | 36 |
| Keyframes accepted/dropped | 36/0 |
| Max camera–LiDAR delta | 2.109 ms |
| P95 camera–LiDAR delta | 2.035 ms |
| Max inter-camera spread | 0.000 ms |
| Ảnh Task 2, 6 camera | 216 |

General-purpose annotation xuất thành công `6,101` segment: `1,923 surface_ann`, `4,178 object_ann`; 3D reference bridge tạo `5,249 sample_annotation`. Contract upstream và METEOR factory đều `PASS` trên 36 frame.

### 13.2 Kết quả global

| Metric | Dataset run 2 |
|---|---:|
| Image Drivable IoU ↑ | **0.8719** |
| Image Lane F1 ↑ | **0.3631** |
| Image Lane F1 raw ↑ | 0.3637 |
| Empty lane prediction rate ↓ | **29.63%** |
| BEV Drivable IoU ↑ | **0.1716** |
| BEV Lane F1 ↑ | **0.2090** |
| Fused BEV Lane Chamfer ↓ | **3.366 m** |
| Mask2Former direct inference | 102.94 ms/image |
| Class filter + post-processing | 7.78 ms/image |
| IPM + six-camera fusion | 422.17 ms/frame |

Metric status là `PASS_PROXY_ONLY`: image reference `seg2d21` vẫn có nguồn từ cùng upstream Mask2Former checkpoint. Các số IoU/F1 đo tính nhất quán của pipeline và không thay thế accuracy trên human-reviewed ground truth.

### 13.3 Breakdown theo camera

| Camera | Drive IoU | Lane F1 | Empty rate | Components/frame | Mean conf. | BEV Drive IoU | BEV Chamfer |
|---|---:|---:|---:|---:|---:|---:|---:|
| FRONT_WIDE | 0.8802 | **0.7519** | **5.56%** | 4.944 | 0.931 | 0.3128 | 2.593 m |
| FRONT_LEFT | 0.8812 | 0.2480 | 41.67% | 0.778 | 0.539 | **0.5095** | 1.801 m |
| FRONT_RIGHT | 0.8539 | 0.4066 | 27.78% | 1.917 | 0.701 | 0.3423 | 2.776 m |
| BACK_WIDE | 0.8565 | **0.1004** | 22.22% | 2.694 | 0.758 | 0.1130 | 1.951 m |
| BACK_LEFT | **0.9085** | 0.3790 | 30.56% | 1.389 | 0.677 | 0.1676 | **1.104 m** |
| BACK_RIGHT | 0.8511 | 0.2929 | **50.00%** | 1.556 | 0.473 | 0.1701 | 2.510 m |

`FRONT_WIDE` giữ lane signal ổn định nhất. `BACK_RIGHT` có empty rate cao nhất, còn `BACK_WIDE` có F1 thấp dù không rỗng nhiều; trường hợp này cho thấy mask tồn tại nhưng hình dạng/overlap với proxy reference yếu. Hai camera này cần được kiểm tra trước khi tăng camera fusion weight.

### 13.4 So sánh độ ổn định giữa hai dataset

Đây là so sánh **khác scene**, dùng để phát hiện domain shift; không phải benchmark hai kiến trúc trên cùng dữ liệu.

| Metric | Dataset 1 `pnk_295` | Dataset 2 `pnk_132` | Delta run 2 − run 1 |
|---|---:|---:|---:|
| Image Drivable IoU ↑ | 0.8477 | **0.8719** | +0.0242 |
| Image Lane F1 ↑ | **0.8710** | 0.3631 | -0.5078 |
| Empty lane rate ↓ | **8.69%** | 29.63% | +20.94 điểm % |
| BEV Drivable IoU ↑ | **0.2417** | 0.1716 | -0.0701 |
| BEV Lane F1 ↑ | **0.3685** | 0.2090 | -0.1595 |
| Fused BEV Chamfer ↓ | 7.578 m | **3.366 m** | -4.213 m |

Lane branch không ổn định giữa hai cảnh: Lane F1 giảm mạnh và empty rate tăng hơn 20 điểm phần trăm. Chamfer lại giảm vì output ở run 2 thưa hơn và các lane còn lại nằm gần reference hơn; vì vậy không được đọc Chamfer riêng lẻ như một cải thiện tổng thể.

### 13.5 Đánh giá bottleneck

- **Segmentation/domain:** là tín hiệu rủi ro chính của run 2. Mức giảm Lane F1 và tăng empty rate tập trung ở side/rear cameras. Do proxy cùng nguồn, cần human GT hoặc LiDAR paint checklog riêng cho scene này để kết luận accuracy, đặc biệt ở 40–70 m.
- **IPM/ground plane:** vẫn là bottleneck rõ. Image Drivable IoU đạt `0.8719` nhưng fused BEV Drivable IoU chỉ `0.1716`; BEV Lane F1 cũng chỉ `0.2090`.
- **Calibration:** algebraic round-trip có sai số gần machine precision, nhưng chỉ xác nhận phép biến đổi thuận/nghịch trong code. Nó chưa chứng minh extrinsic vật lý đúng trên mặt đường ban đêm.
- **Fusion:** fused Chamfer `3.366 m` cao hơn mean available per-camera Chamfer `2.122 m`. Contribution overlap và camera weight vẫn làm giảm geometry ở fused output.

Kết luận cho scene này:

```text
Pipeline chạy hết folder và UFLD-v2 không nằm trong inference path.
Drivable image-space ổn định, lane segmentation yếu hơn rõ rệt so với dataset 1.
BEV tiếp tục chịu ảnh hưởng của ground-plane IPM, calibration và fusion.
Chưa có evidence độc lập 40–70 m cho scene 2, nên không tuyên bố accuracy far-range.
```

### 13.6 Deliverables

```text
out/task2_lane_bev/pnk_132_1772278707_1772278713/mask2former_unified/
├── validation_report.json              # PASS_PROXY_ONLY, failures=0
├── metrics/summary.json
├── metrics/camera_summary.csv
├── artifacts/per_camera/               # 216 artifact sets
├── artifacts/fused_bev/                # 36 fused BEV artifacts
└── deliverables/
    ├── frames/                          # 36 frames, 2560x1440
    └── task2_mask2former_unified_6cam_bev.mp4
```

Video đã mở kiểm tra bằng OpenCV: 36 frame, 2560×1440, 5 FPS, thời lượng 7.2 giây. Frame đầu và cuối đều có đủ 6 camera, overlay drivable/lane và fused BEV ở toàn bộ cột phải.

## 14. Batch 2 - metrics and checklog

### 14.1 Aggregate metrics

| Architecture | Drivable IoU | Image Lane F1 | Empty lane rate | BEV Drivable IoU | BEV Lane F1 | Fused BEV Chamfer (m) | Lane runtime (ms/image) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mask2Former-Small + UFLD-v2 ResNet-18 baseline | 0.6883 | 0.0252 | 91.20% | 0.2213 | N/A | 10.679 | 19.40 |
| Mask2Former-Small + UFLD-v2 ResNet-18 | 0.6883 | 0.0214 | 92.13% | 0.2213 | 0.0068 | 12.064 | 35.06 |
| Mask2Former-Small + CLRNet | 0.6883 | 0.0361 | 92.59% | 0.2213 | 0.0189 | 15.837 | 33.35 |
| Mask2Former-Small + UFLD-v2 ResNet-34 | 0.6883 | 0.0336 | 79.63% | 0.2213 | 0.0110 | 15.298 | 34.70 |
| Mask2Former Swin-L Mapillary unified | 0.8719 | 0.3631 | 29.63% | 0.1716 | 0.2090 | 3.366 | 102.94* |

*Unified runtime is sampled Mask2Former model inference only; its reported post-processing adds 7.78 ms/image. Drivable metrics for the first four rows come from the shared Mask2Former-Small branch.

### 14.2 Per-camera lane metrics

| Camera | UFLD-v2 R18 F1 | Empty | CLRNet F1 | Empty | UFLD-v2 R34 F1 | Empty | Unified F1 | Empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FRONT_WIDE | 0.1200 | 69.44% | 0.2168 | 55.56% | 0.1516 | 63.89% | 0.7519 | 5.56% |
| FRONT_LEFT | 0.0000 | 94.44% | 0.0000 | 100.00% | 0.0164 | 41.67% | 0.2480 | 41.67% |
| FRONT_RIGHT | 0.0000 | 100.00% | 0.0000 | 100.00% | 0.0000 | 100.00% | 0.4066 | 27.78% |
| BACK_WIDE | 0.0087 | 94.44% | 0.0000 | 100.00% | 0.0123 | 91.67% | 0.1004 | 22.22% |
| BACK_LEFT | 0.0000 | 94.44% | 0.0000 | 100.00% | 0.0213 | 86.11% | 0.3790 | 30.56% |
| BACK_RIGHT | 0.0000 | 100.00% | 0.0000 | 100.00% | 0.0000 | 94.44% | 0.2929 | 50.00% |

### 14.3 Checklog

| Validation item | Result |
|---|---:|
| Status | `PASS_PROXY_ONLY` |
| Camera predictions | 216/216 |
| BEV records | 36/36 |
| Rendered frames | 36/36 |
| Failures | 0 |

Metrics compare against Task 1 pseudo-labels. Unified Mask2Former image metrics are source-coupled to its upstream checkpoint; they are consistency metrics, not independent accuracy measurements.
