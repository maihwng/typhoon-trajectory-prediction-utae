# typhoon-trajectory-prediction-utae
Dự án ứng dụng mô hình học sâu UTAE (U-shaped Temporal Attention Encoder) để dự báo tọa độ tâm bão dựa trên dữ liệu khí tượng đa thời gian.
1. Dữ liệu (Data)
Sử dụng dữ liệu tích hợp giai đoạn 2021–2024
:
IBTrACS: Cung cấp tọa độ thực tế (vĩ độ, kinh độ) của các cơn bão
.
ERA5: Các trường khí tượng (áp suất MSLP, nhiệt độ SST, gió U/V, độ ẩm RH...) tại bề mặt và các mực áp suất 200, 500, 850 hPa
.
2. Tiền xử lý (Preprocessing)
Quy trình làm sạch và chuẩn hóa dữ liệu để huấn luyện
:
Chuẩn hóa: Nội suy dữ liệu về bước thời gian cố định 6 giờ/lần và chuẩn hóa kinh độ
.
Đồng bộ: Trích xuất đặc trưng ERA5 quanh tâm bão, đưa về kích thước lưới 96×96
.
Cấu trúc: Dữ liệu đầu vào là chuỗi 5 bước thời gian liên tiếp (T=5) dạng tensor 5 chiều
.
Lưu trữ: Lưu dưới định dạng HDF5 (.h5) để tối ưu tốc độ truy xuất
.
3. Mô hình (Model)
Sử dụng kiến trúc UTAE được điều chỉnh cho bài toán hồi quy tọa độ
:
Spatial Encoder: Các khối tích chập trích xuất đặc trưng không gian đa mức từ dữ liệu khí tượng
.
LTAE (Lightweight Temporal Attention Encoder): Sử dụng cơ chế chú ý (attention) để học mối quan hệ giữa các thời điểm quan sát
.
Khối dự đoán: Lược bỏ Decoder, sử dụng Global Average Pooling và các lớp Fully Connected để dự báo trực tiếp vĩ độ và kinh độ
.
4. Kết quả (Results)
Đánh giá trên tập kiểm thử năm 2024 (29 cơn bão)
:
Sai số khoảng cách trung bình: 17.062 km
.
Độ ổn định: Sai số vĩ độ đạt 0.140°, kinh độ đạt 0.070°
.
Xu hướng: Sai số tăng dần theo thời gian dự báo (Lead step), từ ~4.8 km (6 giờ) lên ~29.2 km (36 giờ)
.
Kết luận: Mô hình hội tụ tốt, không bị hiện tượng quá khớp và nắm bắt hiệu quả động lực học ngắn hạn của bão
.
