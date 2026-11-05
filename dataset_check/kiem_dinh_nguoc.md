## Mục tiêu

Kiểm định tính đúng đắn của 3D ground truth trong dataset LiDAR bằng các ràng buộc hình học và vật lý, không giả định trước convention của label.

Mỗi object label hiện có dạng:

`x, y, z, dx, dy, dz, yaw, class, occlusion, numpoints, trackid`

Các class chính:

- Car
- Rider
- Pedestrian

Cần xác định và kiểm chứng:

- `dx, dy, dz` tương ứng với length / width / height như thế nào.
- `yaw` được đo từ trục nào.
- chiều dương của yaw.
- yaw có offset `90°`, `180°` hay không.
- heading có đúng front/back hay chỉ đúng orientation modulo `180°`.
- `(x,y,z)` là geometric center, bottom center hay convention khác.
- hệ tọa độ và handedness có hợp lý hay không.
- 3D box có thực sự khớp với LiDAR points.
- `numpoints` trong GT có nhất quán với point cloud thực tế hay không.

Không được chỉ dựa vào tên field hoặc exporter code. Phải kiểm định bằng dữ liệu thực tế.

---

## 1. Kiểm tra dimension bằng kích thước vật lý

Thống kê `dx, dy, dz` riêng theo từng class.

Tính ít nhất:

- mean
- median
- P5
- P25
- P75
- P95

Đặc biệt với `Car`, xác định dimension nào có phân bố tương ứng với:

- length: khoảng 3–6 m
- width: khoảng 1.5–2.5 m
- height: khoảng 1.3–2.5 m

Các khoảng trên chỉ dùng làm sanity check, không dùng để sửa GT trực tiếp.

Ví dụ:

- nếu `median(dx) ≈ 4.3`, `median(dy) ≈ 1.8`, `median(dz) ≈ 1.6` thì candidate mạnh là `dx=length, dy=width, dz=height`.
- nếu `median(dx) ≈ 1.8`, `median(dy) ≈ 4.3`, `median(dz) ≈ 1.6` thì candidate mạnh là `dx=width, dy=length, dz=height`.

Dùng thêm Pedestrian để xác định dimension đứng. Dimension có giá trị khoảng chiều cao người thường là height.

Report rõ candidate convention và các object bất thường về kích thước.

---

## 2. Kiểm tra orientation của box bằng LiDAR point cloud

Đối với mỗi box có đủ LiDAR point, lấy các point nằm trong hoặc gần object và xét trong mặt phẳng BEV `(x,y)`.

Ưu tiên:

- class `Car`
- `numpoints >= 20` hoặc một threshold hợp lý khác
- loại bỏ object quá xa hoặc quá sparse

Dùng PCA trên `(x,y)` của object point cloud.

Vector principal axis lớn nhất đại diện cho trục dài hình học của object:

`theta_pca = atan2(v_y, v_x)`

So sánh với yaw GT.

Do PCA không phân biệt đầu/đuôi nên phải tính sai số modulo `π`:

`error = angular_distance_mod_pi(theta_gt, theta_pca)`

Nếu convention đúng, distribution của error phải tập trung gần `0°`.

Nếu error tập trung gần:

- `0°`: yaw phù hợp với long axis.
- `90°`: khả năng cao length/width đang swap hoặc yaw đang biểu diễn width-axis.
- giá trị lớn/phân bố rộng: có thể sai sign, sai coordinate frame, box-point association hoặc GT không đúng.

Xuất histogram và percentile của PCA yaw error.

---

## 3. Brute-force các yaw/dimension convention

Không đoán convention bằng mắt.

Thử tối thiểu các yaw candidate:

- `yaw`
- `-yaw`
- `yaw + π/2`
- `yaw - π/2`
- `-yaw + π/2`
- `-yaw - π/2`

Kết hợp với:

- `dx = length, dy = width`
- `dx = width, dy = length`

Với mỗi candidate, tính PCA orientation error trên tập Car chất lượng tốt.

Report:

- mean error
- median error
- P90 error
- tỷ lệ object có error `< 5°`
- tỷ lệ `< 10°`
- tỷ lệ `< 20°`

Convention tốt nhất phải cho error nhỏ một cách có hệ thống trên số lượng lớn object.

Không kết luận dựa trên vài sample.

---

## 4. Kiểm tra heading front/back bằng temporal motion

PCA chỉ xác định orientation modulo `180°`, không xác định đầu xe.

Dùng `trackid` để kiểm tra heading thực sự.

Với cùng một object qua nhiều frame:

`motion = center(t+k) - center(t)`

Tính:

`theta_motion = atan2(delta_y, delta_x)`

So sánh với yaw GT:

`delta = wrap_to_pi(theta_yaw - theta_motion)`

Chỉ sử dụng object có displacement đủ lớn để tránh nhiễu, ví dụ:

- displacement > 1 m
- hoặc tốc độ vượt threshold hợp lý

Ưu tiên Car.

Nếu đa số moving Car có:

- `delta ≈ 0°`: heading phù hợp với hướng chuyển động.
- `delta ≈ 180°`: orientation đúng nhưng front/back có khả năng bị đảo.
- `delta ≈ ±90°`: yaw convention hoặc length/width có khả năng sai.

Không dùng object đứng yên để đánh giá heading.

Report histogram `yaw - motion heading`.

---

## 5. Kiểm tra center Z và ground contact

Xác định `z` trong GT là geometric center hay bottom center.

Ước lượng ground height gần object từ LiDAR local neighborhood.

Thử các giả thuyết:

### Candidate A: geometric center

`z_bottom = z - height / 2`

### Candidate B: bottom center

`z_bottom = z`

So sánh `z_bottom` với estimated ground plane.

Đối với Car, Pedestrian, Rider, bottom của box phải nằm gần mặt đường trong phần lớn sample.

Report cho mỗi convention:

- median `|z_bottom - z_ground|`
- P90 error
- phân bố error
- tỷ lệ object có ground error dưới threshold, ví dụ 0.2 m hoặc 0.3 m

Convention nào cho bottom box bám mặt đất tốt hơn trên toàn dataset là candidate hợp lý hơn.

Đồng thời dùng kiểm tra này để xác nhận dimension nào là height.

---

## 6. Recompute số LiDAR point trong box và so với `numpoints`

Với từng candidate convention, dựng lại oriented 3D box và đếm số LiDAR point nằm bên trong.

Tính:

`num_points_computed`

So sánh với field:

`numpoints_gt`

Report:

- Pearson correlation
- Spearman correlation
- mean absolute error
- median absolute error
- tỷ lệ box có exact match nếu có thể
- scatter plot `numpoints_gt` vs `numpoints_computed`

Nếu `numpoints` được tạo trực tiếp từ annotation box, convention đúng phải cho mức nhất quán rất cao.

Đây là một trong các kiểm tra mạnh nhất để xác nhận:

- center convention
- dimension order
- yaw convention
- point coordinate frame

Nếu tất cả convention đều cho correlation thấp thì cần kiểm tra liệu `numpoints` được tính từ sensor khác, filtered cloud hoặc accumulated cloud.

---

## 7. Kiểm tra mức độ box bao phủ object point cloud

Ngoài `numpoints`, tính các metric hình học:

- tỷ lệ point object nằm trong box
- khoảng cách point đến box surface
- số point nằm sát ngoài box
- occupancy theo chiều dài, rộng, cao

Một box hợp lý không nên:

- cắt xuyên qua phần lớn cluster
- quá nhỏ khiến nhiều point nằm ngoài
- quá lớn một cách hệ thống
- có long axis vuông góc với object point cloud

Flag các trường hợp có geometry bất thường để inspect thủ công.

---

## 8. Kiểm tra tính liên tục theo track

Với mỗi `trackid`, kiểm tra theo thời gian:

- center displacement
- dimensions
- yaw
- velocity
- acceleration

Dimensions của cùng một object không nên nhảy mạnh giữa các frame.

Yaw cũng không nên thay đổi đột ngột khi xe đang chuyển động ổn định.

Flag nếu:

- length/width thay đổi bất thường
- center teleport
- yaw flip `≈180°` liên tục giữa các frame
- yaw jump lớn nhưng motion direction gần như không đổi
- velocity hoặc acceleration phi vật lý

Mục đích là phát hiện lỗi annotation hoặc lỗi parsing/convention.

---

## 9. Kiểm tra coordinate system và handedness

Dùng trajectory của moving objects để xác định yaw sign và coordinate handedness.

Kiểm tra xem công thức:

`heading_vector = [cos(yaw), sin(yaw)]`

có cùng hướng với motion vector hay không.

Nếu không, thử các candidate:

- `[cos(yaw), sin(yaw)]`
- `[cos(yaw), -sin(yaw)]`
- `[sin(yaw), cos(yaw)]`
- `[-sin(yaw), cos(yaw)]`

Mục tiêu là xác định:

- yaw tính từ +X hay +Y
- yaw tăng clockwise hay counter-clockwise
- hệ tọa độ right-handed hay convention khác

Phải đưa ra kết luận bằng thống kê toàn dataset, không bằng một vài visualization.

---

## 10. Cross-check bằng camera nếu calibration tồn tại

Nếu có LiDAR-to-camera extrinsic và camera intrinsic, project 8 corner của 3D box lên 6 camera.

Không cần có 2D semantic GT.

Kiểm tra định tính:

- box có phủ đúng object trong ảnh không
- bottom box có nằm gần bánh xe/chân object không
- orientation box có hợp lý không
- một object xuất hiện ở nhiều camera có projection nhất quán không

Chọn khoảng 50–100 case đại diện:

- gần
- xa
- phía trước
- phía sau
- bên trái/phải
- nhiều yaw khác nhau
- Car/Pedestrian/Rider
- object có nhiều LiDAR point
- các outlier do automatic audit phát hiện

Xuất visualization overlay để inspect.

Camera chỉ là cross-check bổ sung, không dùng một mình để kết luận convention.

---

## 11. Thứ tự ưu tiên kiểm định

Thực hiện theo thứ tự:

1. Dimension statistics.
2. Ground / Z convention.
3. PCA long-axis validation.
4. Brute-force yaw + dimension convention.
5. Recompute points-inside-box và so với `numpoints`.
6. Temporal motion để xác định heading front/back.
7. Track consistency.
8. Camera projection sanity check.

---

## 12. Kết quả cuối cùng cần trả về

Agent phải tạo một report kết luận rõ:

### Dimension convention

Ví dụ:

`dx = length, dy = width, dz = height`

hoặc convention khác.

### Center convention

Ví dụ:

`x,y,z = geometric center`

hoặc:

`z = bottom center`

### Yaw convention

Phải mô tả rõ:

- yaw tính từ axis nào
- clockwise hay counter-clockwise
- có cần đổi sign không
- có cần cộng/trừ `π/2` không

Ví dụ:

`heading = yaw measured counter-clockwise from +X`

hoặc công thức chuyển đổi chính xác nếu raw GT dùng convention khác.

### Heading validity

Kết luận:

- heading đúng front/back
- hoặc chỉ đúng orientation modulo `π`
- hoặc bị flip `180°`

### Confidence

Cho từng kết luận:

- HIGH
- MEDIUM
- LOW

và ghi metric hỗ trợ kết luận.

---

## 13. Bảng so sánh convention

Tạo bảng ít nhất gồm:

| Candidate | Median PCA Error | P90 PCA Error | Point-count Corr | Ground Error | Motion Heading Error |
|---|---:|---:|---:|---:|---:|

Rank các candidate từ tốt nhất đến kém nhất.

Convention cuối cùng phải là convention cho kết quả nhất quán nhất đồng thời với:

- kích thước vật lý
- LiDAR geometry
- ground contact
- point count
- temporal motion

Không chọn convention chỉ vì một metric tốt.

---

## 14. Output artifacts

Xuất:

- `audit_summary.md`
- `dimension_stats.csv`
- `convention_comparison.csv`
- `pca_yaw_error.csv`
- `pointcount_validation.csv`
- `track_motion_validation.csv`
- danh sách các box/frame bất thường
- histogram/plot cần thiết
- một thư mục visualization chứa các sample BEV và camera projection nếu có calibration

Trong `audit_summary.md`, ghi ngắn gọn:

1. GT convention được suy ra.
2. Các bằng chứng định lượng.
3. Các điểm chưa chắc chắn.
4. Các annotation lỗi/outlier phát hiện được.
5. Công thức chính xác để convert raw GT sang convention chuẩn được đề xuất.

Nguyên tắc chính:

**Không tin metadata hay tên biến. Hãy tìm convention làm cho kích thước vật thể, point cloud geometry, mặt đất, temporal motion và `numpoints` đồng thời nhất quán về mặt vật lý.**