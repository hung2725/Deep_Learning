# DEEP LEARNING

- **Sinh viên thực hiện:** Phạm Thế Hùng  
- **MSSV:** 2374802010164  
- **Môn học:** Giới Thiệu Học Sâu  
- **Giảng viên:** Nguyễn Thái Anh

## ANN

### Công nghệ sử dụng
- **Python 3**  
- **PyTorch** (torch.nn, torch.optim)  
- **NumPy & Matplotlib**  
- **Scikit-learn** (train_test_split)


## Cách hoạt động 

### Tạo dữ liệu giả lập
- **Lớp 0 (Vòng tròn):** Các điểm dữ liệu nằm gọn bên trong tâm.  
- **Lớp 1 (Vành đai):** Các điểm dữ liệu bao quanh lớp 0.

Dữ liệu sau khi tạo được:
- Chuyển sang dạng **Tensor** để PyTorch xử lý.
- Chia thành **80% tập huấn luyện (train)** và **20% tập kiểm tra (test)**.

### Xây dựng mô hình ANN

- **Cấu trúc mạng:**  
  - Các lớp tuyến tính (`nn.Linear`).  
  - Hàm kích hoạt **ReLU** cho các lớp ẩn.  
  - Hàm **Sigmoid** ở lớp đầu ra để dự đoán xác suất thuộc lớp 0 hoặc 1.

- **Hàm forward:**  
  Dữ liệu được truyền theo thứ tự:
  
  `Input -> Linear -> ReLU -> Linear -> Sigmoid`

---

### Huấn luyện và tối ưu

- **Hàm mất mát (Loss Function):**  
  - `BCELoss`  
  - `BCEWithLogitsLoss` (ổn định số học tốt hơn)

- **Bộ tối ưu (Optimizer):**  
  - `Adam`  
  - `SGD`

- **Quy trình huấn luyện:**
  1. Dự đoán đầu ra từ mô hình.
  2. Tính loss.
  3. Lan truyền ngược (`loss.backward()`).
  4. Cập nhật trọng số (`optimizer.step()`).

## Bài tập về nhà

### Phần 1:

**Cách thực hiện:**
- Tăng số nơ-ron lớp ẩn từ **4 -> 8**.
- Thử nghiệm mạng sâu hơn với **2 lớp ẩn (8 nơ-ron + 6 nơ-ron)**.

**Kết quả:**
- Mạng **8 nơ-ron** hội tụ nhanh và cho kết quả ổn định trên tập dữ liệu này.
- Mạng **8 + 6 nơ-ron** quá phức tạp, dẫn đến loss dao động mạnh trong 100 epoch.

### Phần 2:

**Cách thực hiện:**
- So sánh `BCELoss` và `BCEWithLogitsLoss`.
- So sánh **Adam** với **SGD (learning rate = 0.01)**.

**Kết quả:**
- **Adam** vượt trội về tốc độ giảm loss và khả năng hội tụ.
- **SGD** học rất chậm, loss gần như không thay đổi sau 100 epoch.

---

### Phần 3:

**Cách thực hiện:**
- Lưu loss theo epoch vào các danh sách:
  - `loss_base`
  - `loss_8nodes`
  - `loss_sgd`
- Vẽ các đường loss trên cùng một đồ thị bằng **Matplotlib**.

**Kết quả:**
- Đường **Adam (8 nơ-ron)** giảm nhanh nhất -> hiệu quả tốt nhất.
- Đường **SGD** gần như nằm ngang -> giảm loss chậm nhất.
- Các mô hình dùng **Adam** có dao động nhẹ ở những epoch đầu do cơ chế điều chỉnh tốc độ học thích nghi.

