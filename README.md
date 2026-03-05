# DEEP LEARNING

- **Sinh viên thực hiện:** Phạm Thế Hùng  
- **MSSV:** 2374802010164  
- **Môn học:** Giới Thiệu Học Sâu  
- **Giảng viên:** Nguyễn Thái Anh

## BÀI TẬP QUA TẾT - ANN2

### Công nghệ sử dụng
- **Python 3**
- **PyTorch** (torch.nn, torch.optim)
- **Torchvision**
- **NumPy & Matplotlib**

## Phần 1: Tập dữ liệu MNIST

### Cách hoạt động
- **Chuẩn bị dữ liệu:** Tải bộ ảnh chữ số viết tay MNIST (60,000 ảnh train, 10,000 ảnh test) và chuyển sang dạng **Tensor** để PyTorch xử lý.
- **Xây dựng mô hình ANN:**
  - **Đầu vào:** Biến ảnh 28x28 thành chuỗi 784 nơ-ron.
  - **Lớp ẩn:** 1 lớp 128 nơ-ron dùng hàm kích hoạt **ReLU**.
  - **Lớp đầu ra:** 10 nơ-ron tương ứng với 10 chữ số (0-9).
- **Huấn luyện và tối ưu:**
  - **Hàm mất mát:** `CrossEntropyLoss`.
  - **Optimizer:** `Adam` (learning rate = 0.01).
  - Đầu ra được dự đoán, tính loss, lan truyền ngược (`loss.backward()`) và cập nhật trọng số (`optimizer.step()`) trong 50 epoch.

### Kết quả
- Mô hình hội tụ tốt. Biểu đồ vẽ bằng Matplotlib cho thấy Loss giảm và Accuracy tăng dần.
- Độ chính xác trên tập test đạt **90.90%**.

## Phần 2: Tập dữ liệu Cat and Dog (Chó và Mèo)

### Cách hoạt động
- **Chuẩn bị dữ liệu:** Đọc ảnh từ thư mục chứa ảnh chó mèo, cắt cho đều nhau về kích thước **64x64 pixel** và chuyển sang dạng **Tensor**.
- **Xây dựng mô hình ANN:**
  - **Đầu vào:** Duỗi ảnh thành chuỗi 12,288 nơ-ron (64x64x3 kênh màu).
  - **Lớp ẩn 1:** 128 nơ-ron dùng hàm **ReLU**.
  - **Lớp ẩn 2:** 64 nơ-ron dùng hàm **ReLU**.
  - **Lớp đầu ra:** 1 nơ-ron dùng hàm **Sigmoid** để phân loại chó (0) hoặc mèo (1).
- **Huấn luyện và tối ưu:**
  - **Hàm mất mát:** `BCELoss`.
  - **Optimizer:** `Adam` (learning rate = 0.001).
  - Huấn luyện mô hình trong 100 epoch và lưu lại loss.

### Kết quả
- Việc nhận diện chó mèo khó khăn hơn nên loss giảm với tốc độ chậm hơn so với nhận diện chữ số.
- Độ chính xác trên tập test chỉ đạt **61.54%** (do cấu trúc ANN rất cơ bản này thường chưa đủ mạnh để giải quyết các cấu trúc hình ảnh phức tạp như chó mèo trong thực tế).
