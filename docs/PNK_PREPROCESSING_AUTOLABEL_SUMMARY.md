# PNKData preprocessing → autolabel pipeline: summary

## Mục tiêu và phạm vi

Tài liệu này tổng hợp quá trình từ kiểm tra PNKData, xác định các rủi ro của dữ liệu, preprocessing, đến tạo general-purpose annotations và đưa dữ liệu qua METEOR Autolabel Factory.

CoMET không chạy trong repository này vì không có implementation/source code để sử dụng. Nhánh thay thế dùng Mask2Former Swin-L cho nhãn ảnh 2D, adapter T4Dataset và METEOR Autolabel Factory. Lượt tích hợp mới nhất dùng **LIDAR_TOP**; pilot cũ dùng **LIDAR_MERGED_EGO** và cần được xem là một lượt thử nghiệm riêng.

```text
PNK raw sensors + reference 3D labels
    → LIDAR_TOP preprocessing / sync audit / derived heading sidecar
    → Mask2Former Swin-L image annotations + T4Dataset adapter
    → METEOR Autolabel Factory
    → occupancy-flow export + train smoke + frame visualizations / MP4
```

Đây là pipeline sinh **pseudo-label tổng quát** và kiểm tra khả năng đi hết luồng. Nó chưa chứng minh chất lượng của một mô hình 3D detector, cũng chưa chứng minh đủ chất lượng cho mọi task-specific label của METEOR.

## 1. Đã inspect được gì trong dataset

- Dữ liệu thô nằm dưới `D:\mentor_data\PNKData`; metadata, calibration và label nằm dưới `D:\mentor_data\PNKData_meta_fresh`.
- Dataset có 8 camera pinhole, 4 fisheye, `LIDAR_TOP` 10 Hz, `LIDAR_MERGED_EGO` 5 Hz, NAV/IMU và các sensor khác. Lượt tích hợp hiện tại dùng 8 camera pinhole và `LIDAR_TOP`; **không chạy 12 camera**, không dùng fisheye trong lượt này.
- Calibration đang dùng là `Calib_2`. Tài liệu kiểm định cảnh báo không dùng thư mục `Calib` cũ. Dù vậy, quy ước extrinsic và căn chỉnh tổng thể vẫn cần được xác nhận thêm bằng overlay.
- CSV metadata có 1.924 dòng: 220 dòng được đánh dấu có 3D annotation và 1.703 dòng `num_boxes=0` được đánh dấu chưa annotate. Đây là trạng thái metadata, không nên diễn giải mọi dòng 0 boxes thành một cảnh được xác nhận là không có vật thể.
- Audit toàn bộ label text tìm thấy 23.401 file, 1.353.970 box, 76 file rỗng và không có dòng sai định dạng. Các class là `Car`, `Rider`, `Pedestrian`.
- Label có box center, kích thước, yaw, class, track ID và `numpoints`. Các thống kê hỗ trợ `dx=length`, `dy=width`, `dz=height`; `z` nhiều khả năng là geometric center. Kết luận này dựa trên phân bố kích thước, point-count agreement và ground proxy, không phải đo ground survey độc lập.
- Annotation thưa hơn `LIDAR_TOP`: label/segment được ghi ở khoảng 5 Hz, trong khi TOP là 10 Hz. Lượt tích hợp lấy các anchor có reference label; không coi mọi scan TOP là có GT.

Tham khảo: [Dataset Overview](DATASET_OVERVIEW.md), [3D GT reverse audit](../dataset_check/audit_summary.md), [hướng dẫn kiểm định ngược](../dataset_check/kiem_dinh_nguoc.md).

## 2. Các vấn đề dữ liệu đã phát hiện

### Calibration và hệ tọa độ

- `Calib_2` là bộ calibration được chọn, nhưng phép chiếu chưa được xác nhận hoàn toàn trên toàn bộ camera và thời gian. Audit cũ có 803 trường hợp projection cực đoan bị loại khỏi thống kê; một số overlay vẫn lệch hoặc tạo cạnh box bất thường.
- Chưa xác nhận GNSS antenna lever arm sang ego/LiDAR. Điều này có thể ảnh hưởng vận tốc tâm object suy ra từ GPS khi xe rẽ.
- Không áp dụng lại point deskew cho `LIDAR_MERGED_EGO`: chưa xác định export này đã deskew sẵn hay chưa. Nhánh chạy mới chọn `LIDAR_TOP`, nơi có `point_timestamp`, và deskew riêng.

### Timestamp và chất lượng camera

- Có camera frame drop, khoảng thời gian lấy mẫu không đều và các đoạn nghi freeze. Ở session chạy tích hợp, preprocessing ghi nhận 9 dropout intervals và 1.450 frame ứng viên nghi freeze. Trong 688 camera views tại 86 anchor, report đánh dấu 252 view nghi freeze; 436 view dùng được cho cross-modal check.
- Do đó, timestamp gần nhất không đủ để khẳng định ảnh đúng thời điểm; pipeline ghi lại sync status và loại các ảnh nghi freeze khỏi các cross-modal checks.
- Dữ liệu camera, LiDAR, CAN/IMU có tần số khác nhau. Ghép timestamp/interpolate cần giữ giới hạn khoảng cách và không được âm thầm coi dữ liệu lệch thời gian là đồng bộ.

### Heading/yaw và track

- Dataset có trường `yaw`; không có một trường heading front/back riêng đáng tin cậy để dùng trực tiếp. Audit hình học ủng hộ raw yaw là hướng **trục dài modulo 180°** (`box long-axis = yaw`); chỉ riêng kết quả modulo 180° không thể xác định đầu hay đuôi xe.
- World-frame audit thử NAV `Heading` và `Heading2`; candidate tốt nhất trên cùng 10.807 quan sát Car đang chuyển động là **NAV Heading2 + raw box yaw**. Sai số trục modulo 180° median/P90 là 2,20°/11,86°; 94,57% trong 25°. Đây là kiểm tra theo motion, không phải ground truth độc lập cho front/back.
- Audit track ban đầu gắn cờ 698 trường hợp yaw gần flip 180°. Đây là cảnh báo heuristic, không phải 698 lỗi đã xác nhận. Một ví dụ kiểm tra được là Car track 137: object đang chạy liên tục khoảng 4,8–6,6 m/s, yaw raw nhảy gần 179° trong một frame, trong khi trục world và hướng chuyển động ổn định. Ví dụ này xác nhận có ít nhất một flip một-frame; không chứng minh mọi flag đều sai.
- Một ví dụ trước đó bị diễn giải nhầm là object đang chạy nhưng người dùng xác nhận xe đang đỗ. Vì vậy, **không suy heading từ motion cho object đứng yên**. Policy hai bên đã thống nhất là motion chỉ chọn nhánh yaw/yaw+180° cho track thực sự đang chạy; track đỗ chỉ nhận nhánh truyền từ một đoạn moving đáng tin trong cùng track; track đứng yên hoàn toàn để `heading_invalid`.
- Lưu ý về trạng thái code: runner hiện dùng ngưỡng world-speed để chọn moving anchor; chưa có bộ phân loại parking riêng dựa trên nhiều frame đã xác nhận. Do đó, jitter/teleport của center label có thể vượt ngưỡng và bị hiểu nhầm là motion. Số heading valid của run là kết quả qua các gate hiện có, chưa phải cam kết mọi xe đỗ đều đã bị loại đúng.
- Dataset-wide derived audit ghi 10.601 heading chọn trực tiếp từ motion, 7.241 dòng truyền nhánh trong track và 32.481 dòng vẫn invalid. Các số này là kết quả audit suy diễn trên dữ liệu, không phải số annotation đã được xác nhận thủ công. Trong run tích hợp, heading sidecar có 4.659 heading valid và 5.381 Car/Rider heading invalid; pedestrian front/back không được suy ra.

Evidence track 137: [BEV LiDAR yaw-flip sequence](../out/yaw_flip_evidence/track137_segment284_yaw_flip.png), [camera sequence](../out/yaw_flip_evidence/track137_cam_back_wide_yaw_flip_sequence.jpg). Phương pháp và giới hạn: [world-frame yaw audit](../dataset_check/world_yaw_audit/world_yaw_summary.md).

## 3. Ego compensation và xử lý yaw: cụ thể đã làm gì

### 3.1 Deskew LiDAR theo chuyển động của xe

Đây là **ego-motion compensation cho điểm LiDAR**: sửa hình dạng scan do xe và sensor di chuyển trong lúc LiDAR quay. Nó không tự bù chuyển động riêng của từng xe/vật thể trong cảnh.

1. Dùng `Calib_2` để đổi point từ `LIDAR_TOP` sang ego frame.
2. Đọc timestamp riêng của mỗi point; nội suy NAV tại timestamp đó. NAV position được đưa sang local ENU; `Heading2` đổi từ compass heading sang ENU yaw theo `90° − Heading2`; pitch/roll cũng được nội suy.
3. Đưa từng point từ ego pose tại thời điểm đo sang ENU/world, rồi biến đổi ngược về ego pose tại cuối scan:

   `p_world = R_world_ego(t_point) · p_ego(t_point) + t_ego_world(t_point)`

   `p_ego_end = R_world_ego(t_end)^T · (p_world − t_ego_world(t_end))`

4. Lưu cloud đã deskew vào `out/.../preprocessing/lidar_top_deskewed_ego/`; cloud nguồn không bị ghi đè. NAV bracket/gap hoặc `Heading2_std > 2°` làm frame/point bị từ chối thay vì âm thầm dùng pose kém tin cậy.
5. Khi label anchor không trùng timestamp cuối scan, dùng NAV để rigidly re-reference cloud từ ego-at-scan-end sang ego-at-label-time. Trong camera projection, mỗi point được đưa tới world rồi về ego pose tại timestamp ảnh.

Run tích hợp đã deskew **170/170** TOP scans và tạo **86** label-time samples. Đây chỉ bù chuyển động của ego vehicle; object động không được biến dạng/bù riêng trong cloud.

### 3.2 Từ box `yaw` đến directed heading

`yaw` nguồn xác định hướng trục dài của box nhưng không luôn cho biết đầu xe hay đuôi xe. Ta xử lý theo hai bước, không tự ý cộng/trừ 90°:

1. **Đưa center của box sang world:** tại mỗi label timestamp, transform `(x,y)` ego bằng NAV position và ego yaw sang local ENU. Gom theo `(segment, class, trackid)` để không nối nhầm ID giữa class/segment.
2. **Tính hướng chuyển động của track:** dùng sai phân center world hai phía lân cận để lấy `(vx, vy)`, speed và `atan2(vy,vx)`. Sai phân chỉ dùng khi hai điểm cách nhau không quá 1 giây.
3. **Đưa box axis sang world:** theo convention candidate tốt nhất, `yaw_world_axis = ego_yaw_world + raw_yaw`. So sánh modulo 180° với motion; modulo 180° đánh giá trục box chứ chưa phân biệt front/back.
4. **Chọn nhánh đầu/đuôi cho moving anchor:** code hiện gate speed **1.5–40 m/s**, ego turn **≤8°/s**, NAV heading std **≤2°**, và sai số giữa box axis với motion **≤25°**. Qua gate thì chọn `yaw_world_axis` hoặc `yaw_world_axis + 180°` gần hướng chuyển động hơn, với giả định xe chạy tiến.
5. **Truyền nhánh trong cùng track:** chia track tại gap lớn hơn **0,75 s**; trong mỗi đoạn có anchor, chọn nhánh 180° gần nhất với frame trước/sau. Từng frame vẫn giữ box axis đo từ frame đó; chỉ truyền dấu front/back. Không có anchor đáng tin thì heading invalid.
6. **Đưa vào output:** heading world hợp lệ được ghi vào rotation của `sample_annotation` trong scene output; sidecar giữ raw yaw, heading, source, confidence và validity. Nếu invalid, output giữ rotation/yaw modulo-180 nguồn. File label `.txt` gốc vẫn read-only.

### 3.3 Những gì người dùng và chúng ta đã hoàn thành

| Người dùng xác định / kiểm tra | Đã triển khai hoặc kiểm tra trong repository |
|---|---|
| Chọn `LIDAR_TOP`, tạm bỏ `LIDAR_MERGED_EGO` để tránh dùng cloud ghép có quy ước deskew chưa rõ. | Runner tích hợp đọc TOP, deskew từng point bằng NAV và `Calib_2`; pilot cũ dùng merged được ghi nhận riêng. |
| Phân biệt box-axis `yaw` với directed heading; không mặc định offset ±90°. | Reverse audit xếp `raw yaw`, `dx=length`, `dy=width` đứng đầu; world audit chọn `Heading2 + raw yaw`. |
| Dùng object motion trong world/global frame để chọn yaw hay yaw+180° khi object thực sự đang chạy. | Audit transform track center sang ENU, ước lượng motion, áp dụng các gate nêu trên, rồi xuất source/confidence/validity. |
| Xe đỗ không được tự gán front/back từ motion; track đỗ chỉ nhận hướng từ đoạn moving cùng track; đứng yên toàn track thì invalid. | Policy được ghi nhận trong report và nhánh propagation chỉ chạy khi track chunk có moving anchor. Tuy vậy, code hiện dùng speed threshold làm moving gate; chưa có parking-state classifier độc lập nên còn nguy cơ false anchor do nhiễu center. |
| Yêu cầu kiểm tra trực quan ví dụ yaw flip thay vì kết luận từ thống kê đơn lẻ. | Tạo BEV và camera evidence cho track 137; một frame yaw đổi gần 179° trong khi box center vẫn di chuyển liên tục và world motion gần như giữ nguyên. |
| Không sửa trực tiếp dataset nguồn. | Preprocessing, heading audit và METEOR scene được ghi dưới `out/`; report đánh dấu raw data/source labels không bị sửa. |

Trong world-frame audit dataset, candidate `Heading2 + raw yaw` đạt median/P90 **2,20°/11,86° modulo 180°** trên 10.807 moving-Car observations. Đây là evidence cho đúng **trục dài**, không phải accuracy front/back. Trong integrated run có **2.704** heading trực tiếp từ motion + **1.955** nhánh truyền trong track = **4.659** resolved; **5.381** Car/Rider invalid và **324** pedestrian không suy front/back. Valid không đồng nghĩa đã được xác minh bằng mắt từng box.

**Phần còn phải hoàn thiện trước khi coi parking/yaw đã giải quyết hoàn toàn:** thêm hoặc kiểm chứng bộ xác nhận trạng thái moving/parked dựa trên displacement ổn định nhiều frame và ngưỡng noise; kiểm tra track đã biết là xe đỗ (bao gồm ví dụ người dùng đã chỉ ra) để bảo đảm không phát sinh motion anchor; chỉ sau đó mới tin số resolved. Giữ `heading_invalid` cho track static-only hoặc motion mơ hồ.

Code chính: [sensor preprocessing](../pnk_comet_autolabel/src/pnk_comet_autolabel/sensor_preprocessing.py), [heading audit của runner](../pnk_top_autolabel/src/pnk_top_autolabel/heading.py), [áp dụng heading vào scene output](../pnk_top_autolabel/src/pnk_top_autolabel/pipeline.py). Artifact: [BEV track 137](../out/yaw_flip_evidence/track137_segment284_yaw_flip.png), [camera track 137](../out/yaw_flip_evidence/track137_cam_back_wide_yaw_flip_sequence.jpg), [world-frame yaw audit](../dataset_check/world_yaw_audit/world_yaw_summary.md).

Chi tiết sensor health và artifact preprocessing: [preprocessing report](../out/pnk_top_e2e_290_1772350499_1772350516/preprocessing/preprocessing_report.md). Deskew scan không đồng nghĩa với motion compensation cho từng object động trong calibration overlay; report vẫn yêu cầu dùng static geometry khi kiểm tra calibration.

## 4. Autolabel pipeline đã chạy

### Nhánh CoMET replacement tích hợp mới

- **2D:** Mask2Former Swin-L pretrained trên Mapillary Vistas chạy trên 8 camera pinhole tại 86 anchor, tổng 688 view. Sinh segmentation/panoptic dùng làm general image labels; METEOR factory kiểm tra segmentation artifacts.
- **3D:** `sample_annotation` lấy từ reference 3D labels PNK và chuyển qua T4Dataset adapter. **Không chạy CenterPoint/MMDetection3D trong lượt này**; vì vậy không có kết quả đo chất lượng 3D detector.
- **Downstream:** METEOR Autolabel Factory tạo/đọc general annotations gồm `surface_ann`, `object_ann`, `sample_annotation`; flow targets được rasterize từ track/box và future displacement. Occupancy flow kế thừa chất lượng của 3D source annotations.
- **Train handoff:** chạy một training smoke để kiểm tra loader/config có thể nhận scene. Report ghi không có validation split; đây không phải train/evaluate đầy đủ và không có metric chất lượng model.
- **Deliverables:** PNG theo frame và MP4 ghép từ kết quả frame. Video là visualization của pipeline output, không phải video được model sinh trực tiếp.

### Pilot cũ cần phân biệt

`pnk_comet_autolabel` là pilot trước đó: 71 `LIDAR_MERGED_EGO` keyframes × 8 pinhole camera, bốn fisheye bỏ qua; 3D boxes vẫn là reference annotation. Đây không phải cùng run với pipeline `LIDAR_TOP` mới và không nên gộp số liệu của hai lượt.

## 5. Kết quả run tích hợp tham chiếu

Run: session `20260301_1429_VF6_01_1772350177_1772350814`, label segment `290_1772350499_1772350516`.

| Hạng mục | Kết quả được ghi nhận |
|---|---:|
| TOP scans deskew thành công | 170 / 170 |
| METEOR samples / label anchors | 86 |
| Camera views infer bằng Mask2Former | 688 / 688 |
| Camera views nghi freeze tại anchors | 252 / 688 |
| `surface_ann` / `object_ann` / `sample_annotation` | 4.695 / 9.132 / 10.364 |
| Occupancy-flow target | 86 frame; 88.325 BEV cells; 2.246 moving tracks nguồn |
| METEOR segmentation validation | pass theo run report |
| Training smoke | hoàn tất; validation split không có |
| Nguồn 3D boxes | PNK reference labels, không phải CenterPoint |

Report chính: [run summary](../out/pnk_top_e2e_290_1772350499_1772350516/reports/run_summary.md). Artifacts chính: [PNG frames](../out/pnk_top_e2e_290_1772350499_1772350516/reports/frames/), [MP4](../out/pnk_top_e2e_290_1772350499_1772350516/reports/pnk_top_lidar_top_e2e.mp4), [T4 contract](../out/pnk_top_e2e_290_1772350499_1772350516/reports/t4_contract.json), [factory segmentation check](../out/pnk_top_e2e_290_1772350499_1772350516/reports/factory_segmentation.json).

## 6. Điều đã chứng minh và điều chưa chứng minh

**Đã chứng minh trong phạm vi run:** raw session có thể đi qua preprocessing `LIDAR_TOP` → Mask2Former 2D labels → T4Dataset general annotations → METEOR factory → occupancy-flow export → training smoke → PNG/MP4. Source raw và label không bị sửa.

**Chưa chứng minh:**

- Chưa có independent 2D ground truth cho session để kết luận IoU/F1 của Mask2Former trên PNK.
- 3D boxes là nhãn có sẵn, không phải inference; không có CenterPoint score/metric.
- Occupancy flow đã được tạo và schema/luồng đã kiểm tra, nhưng chất lượng flow chưa có target độc lập để đánh giá.
- Training smoke chỉ kiểm tra handoff; không có validation set hoặc kết quả train/evaluation có ý nghĩa.
- Chưa chứng minh pipeline tạo đủ và đạt chất lượng cho toàn bộ 12 task-specific METEOR labels. Stop line không được xác nhận trong scope run này.
- Chưa xác nhận calibration, camera freeze, NAV-to-ego lever arm và mọi trường hợp front/back heading trên toàn bộ dataset.
- Chưa chạy đủ 12 camera; run mới dùng 8 pinhole camera, fisheye chưa tham gia.

Vì vậy, nên gọi output hiện có là **general-purpose pseudo-GT / annotations đã qua schema và flow checks**, không gọi là fully validated ground truth.

## 7. Việc còn lại được ưu tiên

1. Xác nhận coordinate/extrinsic conventions và NAV/LiDAR reference bằng overlay tĩnh đại diện; xác minh lever arm nếu cần dùng world velocity chính xác.
2. Cải thiện camera health gate và rà soát các đoạn freeze/dropout, đặc biệt trước khi dùng ảnh làm cross-modal evidence.
3. Giữ policy heading: motion chỉ cho track đang chạy; parked-only track không tự gán front/back; theo dõi riêng `heading_valid`, source và confidence.
4. Tạo một tập kiểm tra thủ công/annotated nhỏ cho image segmentation, 3D orientation và occupancy flow để có metric độc lập.
5. Nếu cần đánh giá 3D autolabel model, thay nguồn `sample_annotation` reference bằng CenterPoint và giữ riêng provenance/metrics của model.
6. Kiểm kê task-specific outputs của METEOR factory; bổ sung các task còn thiếu và validation trước khi tuyên bố đủ 12 tasks.

## Tài liệu và cách chạy

- [Dataset overview](DATASET_OVERVIEW.md)
- [Reverse audit 3D labels](../dataset_check/audit_summary.md)
- [World-frame heading audit](../dataset_check/world_yaw_audit/world_yaw_summary.md)
- [Integrated runner README](../pnk_top_autolabel/README.md)
- [Legacy CoMET replacement pilot README](../pnk_comet_autolabel/README.md)

Chạy lại runner tích hợp theo README: `pnk_top_autolabel/run_full.ps1`. Run mặc định trỏ tới session nêu ở trên; khi đổi bộ dữ liệu, phải truyền đúng raw folder, label segment tương ứng và scene/run name mới để tránh ghép nhầm timestamp hoặc ghi đè outputs.
