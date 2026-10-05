# Image-space Lane/Drivable-Area Detection + IPM to BEV

> Implementation plan cho pilot 6 camera trên scene Task 1: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

> Kết quả benchmark UFLD-v2/CLRNet và checklog 40–70 m: [REPORT_RESULT.md](REPORT_RESULT.md)

## 1. Bài toán

Xây dựng pipeline từ ảnh camera để phát hiện:

- **Lane**
- **Drivable Area**

Sau đó chuyển kết quả từ **Image Space** sang **BEV (Bird's Eye View)** bằng **Inverse Perspective Mapping (IPM)**.

\[
RGB
\rightarrow
Lane/Drivable\ Detection
\rightarrow
IPM
\rightarrow
BEV\ Lane/Drivable\ Map
\]

### Expected Output

\[
M^{BEV}_{lane}, \quad M^{BEV}_{drive}
\]

Deliverable phải hiển thị metrics/evidence cho Mask2Former Small, UFLD-v2, IPM và fused BEV. Mọi score phải ghi rõ nguồn reference là **human-audited GT**, **Task 1 pseudo-GT/proxy** hay **self-consistency**; xem evaluation contract trong [implementation plan](IMPLEMENTATION_PLAN.md#12-evaluation-contract-metrics-và-evidence).

---

## 2. Tech Stack

| Component | Tech Stack | Vai trò | Expected Output |
|---|---|---|---|
| Drivable Area Segmentation | **Mask2Former Swin-Small** | Semantic segmentation, lấy vùng `road/drivable` | Drivable Mask |
| Lane Detection | **Ultra-Fast Lane Detection v2 (UFLD-v2)** | Phát hiện lane line trong ảnh | Lane Points / Lane Curves |
| BEV Projection | **OpenCV + IPM / Homography** | Chuyển từ image coordinate `(u,v)` sang BEV `(X,Y)` | BEV Lane / Drivable Map |
| Camera Geometry | **Intrinsic `K` + Extrinsic `R,t`** | Cung cấp camera geometry cho IPM | Homography Matrix `H` |

---

## 3. Mask2Former Swin-Small

**Mục đích:** phát hiện vùng mặt đường / drivable area.

**Model:**

[Mask2Former Swin-Small Cityscapes](https://huggingface.co/facebook/mask2former-swin-small-cityscapes-semantic)

Pipeline:

\[
RGB
\xrightarrow{Mask2Former}
Semantic\ Map
\rightarrow
Drivable\ Mask
\]

Output:

```text
Drivable Mask
0 = Non-drivable
1 = Drivable
```

---

## 4. Ultra-Fast Lane Detection v2

**Mục đích:** phát hiện các đường lane trong image space.

**Repository:**

[Ultra-Fast-Lane-Detection-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)

Pipeline:

\[
RGB
\xrightarrow{UFLD-v2}
\{(u_i,v_i)\}_{lane}
\]

Output:

- Lane points
- Lane curves
- Lane coordinates trong image space

---

## 5. OpenCV + IPM

**Mục đích:** chuyển Lane / Drivable Area từ ảnh camera sang BEV.

Input geometry:

\[
K,\quad R,\quad t
\]

với giả định mặt đường:

\[
Z=0
\]

Quan hệ giữa ground plane và image:

\[
p_{image}\sim H p_{ground}
\]

IPM thực hiện phép biến đổi ngược:

\[
p_{ground}\sim H^{-1}p_{image}
\]

Tech stack:

[OpenCV](https://opencv.org/)

---

## 6. Pipeline tổng thể

```text
                    Camera RGB
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
     Mask2Former Swin-S       UFLD-v2
              │                   │
              ▼                   ▼
     Semantic Segmentation    Lane Points
              │                   │
              ▼                   │
       Select Road Class          │
              │                   │
              ▼                   ▼
       Drivable Mask        Lane Coordinates
              │                   │
              └─────────┬─────────┘
                        │
               Camera Calibration
                  K + R + t
                        │
                        ▼
                  IPM / Homography
                        │
                        ▼
                BEV Representation
                   ├─ Lane Map
                   └─ Drivable Map
```

## 7. Tóm tắt

\[
\boxed{
Mask2Former\text{-}S
+
UFLD\text{-}v2
+
OpenCV/IPM
}
\]

Trong đó:

\[
Mask2Former \rightarrow Drivable\ Area
\]

\[
UFLD-v2 \rightarrow Lane
\]

\[
IPM \rightarrow Image\ Space \rightarrow BEV
\]
