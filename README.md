# Typhoon Trajectory Prediction using U-TAE

Dự án ứng dụng mô hình học sâu **UTAE (U-shaped Temporal Attention Encoder)** để dự báo tọa độ tâm bão dựa trên dữ liệu khí tượng đa thời gian.

---

## 1. Data (Dữ liệu)

Sử dụng dữ liệu tích hợp trong giai đoạn **2021–2024**:

### IBTrACS
Cung cấp tọa độ thực tế của các cơn bão:

- Vĩ độ (Latitude)
- Kinh độ (Longitude)
- Thời gian quan sát
- Thông tin quỹ đạo bão

### ERA5
Cung cấp các trường khí tượng đa chiều:

- Áp suất mực biển (MSLP)
- Nhiệt độ bề mặt biển (SST)
- Thành phần gió U/V
- Độ ẩm tương đối (RH)
- Các biến khí tượng tại các mực áp suất:
  - 200 hPa
  - 500 hPa
  - 850 hPa

---

## 2. Preprocessing (Tiền xử lý dữ liệu)

Quy trình tiền xử lý dữ liệu phục vụ huấn luyện mô hình:

### Chuẩn hóa dữ liệu

- Nội suy dữ liệu về bước thời gian cố định:

```
6 giờ/lần
```

- Chuẩn hóa kinh độ và các biến đầu vào.

### Đồng bộ dữ liệu

- Đồng bộ thời gian giữa dữ liệu ERA5 và IBTrACS.
- Trích xuất các đặc trưng khí tượng xung quanh tâm bão.
- Đưa dữ liệu về kích thước lưới:

```
96 × 96
```

### Xây dựng dữ liệu đầu vào

Dữ liệu đầu vào được xây dựng dưới dạng chuỗi thời gian gồm:

```
5 bước thời gian liên tiếp (T = 5)
```

với tensor đầu vào 5 chiều.

### Lưu trữ dữ liệu

Dữ liệu sau xử lý được lưu dưới định dạng:

```
HDF5 (.h5)
```

nhằm tối ưu tốc độ truy xuất trong quá trình huấn luyện.

---

## 3. Model (Mô hình)

Sử dụng kiến trúc:

```
UTAE (U-shaped Temporal Attention Encoder)
```

được điều chỉnh cho bài toán hồi quy tọa độ tâm bão.

### Spatial Encoder

- Sử dụng các khối tích chập (Convolutional Blocks).
- Trích xuất đặc trưng không gian đa mức từ dữ liệu khí tượng.
- Học các mẫu phân bố khí tượng xung quanh tâm bão.

### LTAE (Lightweight Temporal Attention Encoder)

- Áp dụng cơ chế chú ý theo thời gian (Temporal Attention).
- Học mối quan hệ giữa các thời điểm quan sát.
- Xác định các thời điểm quan trọng ảnh hưởng đến hướng di chuyển của bão.

### Khối dự đoán

Khác với UTAE gốc sử dụng Decoder cho bài toán phân đoạn ảnh, mô hình trong dự án được điều chỉnh:

- Lược bỏ Decoder.
- Sử dụng Global Average Pooling.
- Kết hợp các lớp Fully Connected.

Đầu ra mô hình:

```
Latitude, Longitude
```

dùng để dự báo trực tiếp tọa độ tâm bão.

---

## 4. Results (Kết quả)

Đánh giá trên tập kiểm thử năm **2024** với:

```
29 cơn bão
```

### Hiệu năng mô hình

| Metric | Value |
|---|---|
| Sai số khoảng cách trung bình | 17.062 km |
| Sai số vĩ độ | 0.140° |
| Sai số kinh độ | 0.070° |

### Đánh giá theo thời gian dự báo (Lead Step)

Sai số tăng dần theo thời gian dự báo:

| Thời gian dự báo | Sai số |
|---|---|
| 6 giờ | ~4.8 km |
| 36 giờ | ~29.2 km |

### Nhận xét

- Mô hình hội tụ tốt trong quá trình huấn luyện.
- Không xảy ra hiện tượng quá khớp nghiêm trọng.
- Có khả năng học được động lực học ngắn hạn của bão.
- Mô hình có hiệu quả trong việc dự báo quỹ đạo bão dựa trên dữ liệu khí tượng đa thời gian.

---

## 5. Technologies

- Python
- Deep Learning
- PyTorch
- UTAE Architecture
- Temporal Attention Mechanism
- ERA5 Dataset
- IBTrACS Dataset
- HDF5
