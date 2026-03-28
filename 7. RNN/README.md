# DEEP LEARNING

- **Sinh viên thực hiện:** Phạm Thế Hùng  
- **MSSV:** 2374802010164  
- **Môn học:** Giới Thiệu Học Sâu  
- **Giảng viên:** Nguyễn Thái Anh  

## RNN — Dự báo chuỗi thời gian bằng mạng nơ-ron hồi quy

### Công nghệ sử dụng

- **Python 3**  
- **PyTorch**  
- **Matplotlib**  
- **NumPy**  
- **pandas**  
- **scikit-learn**  

## Bài tập về nhà

### Bài 1:

**Cách hoạt động:**

- Với bài sóng sin: **100** điểm (`x = np.linspace(0, 20, …)`), `data = sin(x) + 0.1 * nhiễu`, chuẩn hóa Min–Max về `[0, 1]`, tensor `[100, 1]`; cửa sổ `seq_length = 20` tạo 80 mẫu; chia **70%** train, **15%** validation, **15%** test.  
- Với bài đa biến: **300** bước thời gian, ba đặc trưng (sin, cos, nhánh tuyến tính có nhiễu) và **target** tổ hợp có trọng số; đưa vào `pandas.DataFrame`, trực quan hóa; cùng logic chuẩn hóa, `seq_length = 20` và tỷ lệ **70% / 15% / 15%** theo đề.  

**Kết quả:**

- Pipeline tiền xử lý và tập mẫu `(X, y)` đúng kích thước và cách chia tập như trong `RNN.ipynb`; phần đa biến thể hiện rõ cấu trúc dữ liệu qua bảng và đồ thị.  

### Bài 2: 

**Cách hoạt động:**

- Định nghĩa lớp `RNN`: `nn.RNN(input_size=3, hidden_size=32, batch_first=True)` + `nn.Linear(32, 1)`, lấy đầu ra tại bước thời gian cuối; với dữ liệu một kênh, tensor đầu vào được `repeat` thành 3 kênh để khớp `input_size=3`.  
- Huấn luyện với **MSELoss**, **Adam** (`lr=0.01`), `batch_size=16`, **150** epoch; mỗi **10** epoch in **Train Loss** và **Validation Loss**.  

**Kết quả:**

- Tại **epoch 150/150**: **Train Loss 0.0041**, **Validation Loss 0.0016** (đúng stdout trong `RNN.ipynb`). Đường loss train/val theo dõi được qua các mốc epoch 10, 20, …, 150.  

### Bài 3: 

**Cách hoạt động:**

- Đặt mô hình ở chế độ `eval`, dự đoán trên **tập test**; tính **MSE** và **MAE** bằng `sklearn.metrics`; vẽ biểu đồ **Giá trị thực (Actual)** và **Giá trị dự đoán (Predicted)**.  

**Kết quả:**

- Trên tập test: **MSE = 0.0039**, **MAE = 0.0508**. Biểu đồ cho thấy đường dự đoán bám xu hướng sóng.  

### Bài 4: 
**Cách hoạt động:**

- **Trường hợp 1:** Sóng sin 100 điểm (Min–Max), `seq_length = 10`, chia train/test **80% / 20%**; `nn.RNN` với `input_size=1`, `hidden_size=32`, `num_layers=2`, `dropout=0.2`; **Adam** `lr=0.01`, `batch_size=16`, **200** epoch (in loss mỗi 20 epoch).  
- **Trường hợp 2:** Cùng kiểu dữ liệu, `seq_length=20`, `hidden_size=64`, `num_layers=2`, `dropout=0.3`; **Adam** `lr=0.005`, **200** epoch.  

**Kết quả:**
- **Trường hợp 1** (epoch 200): **Train Loss 0.0025**, **Validation Loss 0.0019**.  
- **Trường hợp 2** (epoch 200): **Train Loss 0.0062**, **Validation Loss 0.0032**.  
