# Task 2 — Unified Mask2Former Drivable/Lane BEV

Pipeline hiện tại dùng **một Mask2Former Swin-L Mapillary Vistas Panoptic** để tạo đồng thời drivable area và lane marking cho `71 frame × 6 camera = 426 ảnh`:

```text
Multi-camera RGB
  → Mask2Former Mapillary panoptic/semantic
  → class filtering: drivable mask + lane mask
  → lane confidence/morphology/component filtering
  → calibrated ground-plane IPM
  → camera prior × semantic confidence × distance reliability
  → fused drivable/lane BEV
  → per-camera metrics, artifacts và video
```

CLI riêng `pnk-mask2former-bev` không import hoặc gọi UFLD-v2. Baseline UFLD-v2/CLRNet cũ vẫn còn để benchmark regression.

## Chạy

Từ thư mục gốc repository:

```powershell
pnk_comet_autolabel/.venv/Scripts/python.exe -m pnk_lane_bev.unified_cli run
```

Các stage độc lập:

```powershell
pnk_comet_autolabel/.venv/Scripts/python.exe -m pnk_lane_bev.unified_cli prepare
pnk_comet_autolabel/.venv/Scripts/python.exe -m pnk_lane_bev.unified_cli infer
pnk_comet_autolabel/.venv/Scripts/python.exe -m pnk_lane_bev.unified_cli benchmark
pnk_comet_autolabel/.venv/Scripts/python.exe -m pnk_lane_bev.unified_cli project
pnk_comet_autolabel/.venv/Scripts/python.exe -m pnk_lane_bev.unified_cli evaluate
pnk_comet_autolabel/.venv/Scripts/python.exe -m pnk_lane_bev.unified_cli render
pnk_comet_autolabel/.venv/Scripts/python.exe -m pnk_lane_bev.unified_cli validate
```

Mặc định pipeline tái sử dụng panoptic cache đã sinh bằng đúng checkpoint Mapillary trong Task 1. Khi không truyền `--source-panoptic-cache`, `Mask2FormerDetector` chạy trực tiếp từ RGB và checkpoint cấu hình.

## Cấu hình

File [mapillary_swin_large.yaml](configs/segmentation/mapillary_swin_large.yaml) chứa:

- model/checkpoint và local Hugging Face cache;
- `drivable_classes` và `lane_classes` theo tên taxonomy;
- confidence threshold riêng cho sáu camera;
- morphology, minimum component size và optional skeletonization;
- enable flag và reliability weight cho từng camera;
- threshold của fused BEV.

## Kết quả full folder

| Pipeline | Image Drivable IoU | Lane F1 | Empty lane | BEV Drivable IoU | BEV Lane F1 | BEV Chamfer |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Mask2Former-Small + UFLD-v2 | 0.7690 | 0.0997 | 78.17% | 0.2382 | N/A | 15.216 m |
| Unified Mask2Former Swin-L | **0.8477** | **0.8710** | **8.69%** | **0.2417** | **0.3685** | **7.578 m** |

Trạng thái validation: `PASS_PROXY_ONLY`, với `426/426` prediction, `71/71` BEV và `71/71` video frame.

Image reference `seg2d21` được tạo từ cùng upstream Mask2Former checkpoint. Vì vậy Image IoU/F1 đo class mapping và pipeline consistency, chưa phải accuracy độc lập. BEV vẫn còn gap lớn; IPM/ground-plane, calibration và fusion cần được audit bằng human GT.

## Deliverables

```text
out/task2_lane_bev/pnk_295_1772351165_1772351179/mask2former_unified/
├── cache/mask2former_unified/          # 426 semantic/panoptic + binary masks
├── bev/                                # 71 fused BEV NPZ
├── bev_visual/per_camera/              # BEV riêng của sáu camera
├── artifacts/per_camera/               # RGB, semantic, masks, postprocess, BEV, debug
├── artifacts/fused_bev/                # 71 fused BEV PNG
├── metrics/
│   ├── image_metrics.csv
│   ├── bev_metrics.csv
│   ├── per_camera_bev_metrics.csv
│   ├── camera_summary.csv
│   └── summary.json
├── validation_report.json
└── deliverables/
    ├── frames/                          # 71 ảnh 2560×1440
    └── task2_mask2former_unified_6cam_bev.mp4
```

Kết quả đầy đủ và so sánh với UFLD-v2/CLRNet nằm tại [REPORT_RESULT.md](../docs/Drivable_lane_detection/REPORT_RESULT.md).
