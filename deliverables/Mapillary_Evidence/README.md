# Mask2Former Mapillary evidence

## Is it currently the strongest lane result?

For general lane marking in the evaluations done so far, **yes**. On the fully labeled RLMD validation set it reached lane IoU **26.10%** and F1 **41.39%**. UFLD-v2 ResNet-34 reached F1 **1.40%** on its 100-image validation quarter (different subset and inference raster, so this is indicative rather than a strict matched benchmark). On PNK Batch 2, Mapillary Swin-L reached proxy lane F1 **0.3631**, compared with UFLD-v2 ResNet-34 **0.0336** and CLRNet **0.0361**. PNK references are pseudo-labels and the unified Mask2Former score is source-coupled; treat those as consistency evidence, not independent accuracy.

This does **not** mean it is a reliable stop-line detector: its Mapillary taxonomy has no dedicated stop-line class. Against the stop-line-only RLMD pixels, generic lane output scored IoU **1.43%**, F1 **2.81%**, precision **1.45%**. Drivable IoU cannot be evaluated on RLMD because this split has no drivable-area GT.

## Visual runs

- [RLMD validation contact sheet: 16 GT-positive samples](rlmd_val_contact_sheet.jpg)
- [Browse all 400 RLMD prediction-vs-GT overlays](rlmd_val_gallery.html)
- [Representative RLMD overlay](../../out/mask2former/task_2_rlmd_1080p/overlays/43690.jpg)
- [RLMD metrics and model details](../../docs/Drivable_lane_detection/REPORT_TASK2_MASK2FORMER_MAPILLARY_RLMD.md)
- [Batch 1: 6-camera + fused BEV video](../../deliverables/Task2_All_Batches/batch_1_20260301/mask2former_unified/task2_mask2former_unified_6cam_bev.mp4)
- [Batch 1 sample frame](../../deliverables/Task2_All_Batches/batch_1_20260301/mask2former_unified/frames/0000_1772351165099740.jpg)
- [Batch 2: 6-camera + fused BEV video](../../deliverables/Task2_All_Batches/batch_2_20260228/mask2former_unified/task2_mask2former_unified_6cam_bev.mp4)
- [Batch 2 sample frame](../../deliverables/Task2_All_Batches/batch_2_20260228/mask2former_unified/frames/0000_1772278707099758.jpg)
- [Batch 3 Mapillary lane/drivable overlays (combined with Grounded SAM 2 stop-line branch)](../../out/grounded_sam2/batch_3/overlays/00000_CAM_FRONT_WIDE.jpg)
- [Batch 3 run report](../../docs/Drivable_lane_detection/REPORT_GROUNDED_SAM2_STOPLINE.md)
- [PNK Batch 1/2 metrics, per-camera results, and checklogs](../../docs/Drivable_lane_detection/REPORT_RESULT.md)

Overlay colors on RLMD: red = model-only, green = GT-only, yellow = overlap. In Batch 1/2 video, review the six camera panels and fused BEV on the right.
