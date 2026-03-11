# DEEP LEARNING

- **Sinh viên thực hiện:** Phạm Thế Hùng  
- **MSSV:** 2374802010164  
- **Môn học:** Giới Thiệu Học Sâu  
- **Giảng viên:** Nguyễn Thái Anh

## CNN

### Công nghệ sử dụng
- **Python 3**  
- **PyTorch**
- **Torchvision**
- **Matplotlib**

## Bài tập về nhà

### Bài 1:

**Cách hoạt động:**
- Thay đổi `for epoch in range(5):` thành `for epoch in range(10):`.
- Tính loss trung bình và độ chính xác ở mỗi chu kỳ và in ra cùng đồ thị.

**Kết quả:**
- Độ chính xác tăng đều lên và đạt được khoảng **98.86%**. Mô hình học kỹ hơn dữ liệu.
- Loss giảm dần qua từng epoch nhưng tốc độ giảm có xu hướng chững lại ở nửa sau

### Bài 2:

**Cách hoạt động:**
- Bổ sung tầng `conv3` (`nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)`).
- Chỉnh sửa `self.fc1` để phù hợp với kênh đầu ra (với kích thước `64 * 1 * 1`).
- Cập nhật hàm `forward` để luồng đi qua `pool(relu(conv3))`.

**Kết quả:**
- Độ chính xác cải thiện thêm một chút (đạt **98.91%**).
- Thêm tầng tích chập giúp mô hình hiểu sâu hơn các hình khối phức tạp nâng cao độ hiểu số viết tay.

### Bài 3:

**Cách hoạt động:**
- Kiểm tra sự khác biệt của siêu tham số quá trình luyện với `lr = 0.001` và `lr = 0.1` cho SGD optimizer.

**Kết quả:**
- Với `lr = 0.001`: Mất mát giảm cực kì chậm, độ chính xác trên tập test giảm xuống **97.84%** do tốc độ học quá nhỏ, cần luyện thêm nhiều epoch để mô hình hội tụ tốt hơn.
- Với `lr = 0.1`: Mất mát giảm đều và nhanh trong 3 epoch đầu, sau đó duy trì và hội tụ tại mức rất ổn định. Độ chính xác tập test đạt **98.68%**, không thay đổi quá nhiều so với thời điểm đầu dùng lr mặc định.

### Bài 4:

**Cách hoạt động:**
- Mở rộng hàm hiển thị bằng `conv2_output = torch.relu(model_lr1.conv2(model_lr1.pool(conv1_output)))`.
- Nâng số lượng đồ thị dọc thêm thành 5 bằng `plt.subplot(1, 5, x)`.
- Hiển thị đầu ra Tensor cho `conv2`

**Kết quả:**
- **`conv1`**: Vẫn nhìn ra hình dáng cơ bản
- **`conv2`**: Feature map bị trừu tượng hóa, trở thành các khối nét ít giống ảnh gốc vì đang trích xuất các đặc trưng lớn hơn cho việc phân loại
