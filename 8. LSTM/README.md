# DEEP LEARNING

- **Sinh viên thực hiện:** Phạm Thế Hùng  
- **MSSV:** 2374802010164  
- **Môn học:** Giới Thiệu Học Sâu  
- **Giảng viên:** Nguyễn Thái Anh  

## LSTM

## 1. Công nghệ sử dụng
- **Python**  
- **PyTorch**  
- **Matplotlib**  
- **pandas**  

## 2. Bài 1: Dự đoán giá trị tiếp theo của chuỗi thời gian (Nhiệt độ)
### Cách thức hoạt động
- **Dữ liệu đầu vào:** Sử dụng tập dữ liệu `DailyDelhiClimateTrain.csv` với cột `meantemp` (nhiệt độ trung bình).
- **Tiền xử lý:** 
  - Nhiệt độ được chuyển đổi sang số nguyên (`int`) để có thể sử dụng biểu diễn qua lớp `Embedding`.
  - Dữ liệu được chia thành cửa sổ thời gian (sliding window) với `window_size = 7`. Mô hình sẽ nhìn vào chuỗi nhiệt độ 7 ngày liên tiếp để dự đoán nhiệt độ vào ngày thứ 8.
- **Cấu trúc mô hình (`SimpleLSTM`):** 
  - Lớp `Embedding`: Biến đổi giá trị nhiệt độ (dạng phân loại rời rạc) thành vector nhúng (`embedding_dim = 10`).
  - Lớp `LSTM`: Với `hidden_dim = 20`, giúp ghi nhớ trạng thái từ 7 ngày liền trước.
  - Lớp `Linear (Dense)`: Xuất ra `output_dim = 1` đại diện cho kết quả biểu diễn nhiệt độ kiểu số thực liên tục.
- **Tham số quá trình huấn luyện:** 
  - Thuật toán tối ưu: Adam (`lr = 0.01`).
  - Hàm mất mát: Mean Squared Error (`MSELoss`) vì đây là bài toán hồi quy (regression).
  - Số epochs: 50. Thời gian huấn luyện nhanh, lưu lại mất mát (loss) để theo dõi.

### Kết quả
- Mô hình có xu hướng theo dõi và làm mịn đường đi chung theo hướng dự đoán được nhiệt độ (xu hướng ví dụ giảm/tăng) của chuỗi thời gian.
- Trực quan hóa giá trị thực tế và dự đoán ở 50 ngày cuối cho thấy:
  - **Ưu điểm:** Mô hình đã nắm bắt đúng xu hướng của nhiệt độ.
  - **Nhược điểm:** Kết quả dự đoán còn chậm một nhịp so với thực tế và phản ứng chậm khi thời tiết thay đổi đột ngột (ví dụ thời tiết rơi rớt đột ngột 10°C nhưng dự đoán vẫn đưa ra 14.32°C).

---

## 3. Bài 2: Dự đoán từ tiếp theo của câu ở mức đơn giản
### Cách thức hoạt động
- **Dữ liệu đầu vào:** Bộ dữ liệu tự tạo gồm các câu đơn giản quen thuộc (VD: "tôi thích nghe nhạc", "tôi thích xem phim"...). 
- **Tiền xử lý:**
  - Khởi tạo từ điển `vocab` gồm 16 từ khác nhau bằng cách gán số ID cho từng phần tử (tách từ và tokenization số).
  - Tách câu thành các mảnh (sequence) đầu vào $X$ và từ tiếp theo sẽ làm nhãn $y$.
- **Cấu trúc mô hình (`SimpleLSTM`):**
  - Tương tự như Bài 1, nhưng kết xuất (output) của lớp `Linear` sẽ tương đương định cỡ kích thước từ vựng `output_dim = len(vocab) + 1` để thực hiện bài toán phân lớp đa nhãn.
- **Tham số quá trình huấn luyện:**
  - Thuật toán tối ưu SGD truyền thống với `lr = 0.1`.
  - Hàm mất mát: `CrossEntropyLoss` (đặc trưng cho bài toán bài dự đoán từ tiếp theo/phân loại lớp).
  - Số epochs: 100.
  
### Kết quả 
- Quan sát biểu đồ loss qua các bước (epochs) thấy loss giảm ổn định và hội tụ.
- Dự đoán trên tập thử nghiệm rất chính xác với ví dụ: Mô hình phán đoán từ tiếp theo của cụm `"tôi thích"` sẽ xuất ra chính xác từ thích vị cho ngữ cảnh ví dụ `"xem"` thì tiếp từ có xu hướng là `"phim"`, đúng với ngữ nghĩa được học trong tập huấn luyện nhỏ bé ban đầu.
