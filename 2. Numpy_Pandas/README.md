# DEEP LEARNING

- **Sinh Viên Thực Hiên:** Phạm Thế Hùng
- **MSSV:** 2374802010164
- **Môn Học:** Giới Thiệu Học Sâu
- **Giảng viên:** Nguyễn Thái Anh

## Numpy_Pandas

## Công nghệ sử dụng

- **NumPy**: Thư viện tính toán số học hiệu suất cao
  - Cài đặt: `pip install numpy`
- **Jupyter Notebook**: Môi trường phát triển tương tác

## Cách hoạt động

### Khái niệm cơ bản về NumPy
- NumPy là thư viện mảng (array) thay thế cho list Python thông thường
- **Ưu điểm**: 
  - Tốc độ xử lý nhanh hơn
  - Nhiều hàm tích hợp sẵn cho AI/ML
  - Hỗ trợ tính toán vector hóa

### Tạo và chuyển đổi mảng
- Chuyển đổi Python list sang NumPy array: `np.array()`
- NumPy tự động chuyển đổi kiểu dữ liệu để đồng nhất
- Chuyển đổi kiểu dữ liệu: `.astype()`

### Indexing và Slicing
- **Mảng 1 chiều**: `array[start:end:step]`
- **Mảng 2 chiều**: `array[row, col]` hoặc `array[row_index, col_index]`
- **Fancy Indexing**: Sử dụng list để truy cập nhiều phần tử: `array[[index1, index2]]`

### Boolean Indexing
- Tạo điều kiện boolean: `array % 2 == 0`
- Lọc dữ liệu: `array[condition]`
- Kết hợp nhiều điều kiện: `&` (AND), `|` (OR)


## Bài Tập Về Nhà

### **BTVN `1**

**Mô tả**: Xây dựng game Tic-Tac-Toe sử dụng NumPy array

**Cách hoạt động**:
- Khởi tạo ma trận 3x3 với giá trị 99
- Vòng lặp 9 lượt
- Mỗi lượt: nhập tọa độ (hàng, cột), kiểm tra ô trống
- Sau mỗi lượt: kiểm tra điều kiện thắng
- Kết thúc khi có người thắng hoặc hòa

**Kết quả**: Game chạy tương tác, hiển thị ma trận sau mỗi lượt, thông báo người thắng hoặc hòa

### **BTVN 2**

**Mô tả**: Thực hành các cách truy cập phần tử trong mảng 2D

**Cho mảng**: `y = [[1,2,3], [4,5,6], [7,8,9]]`

**Kết quả**: Hiển thị đúng các phần tử được yêu cầu


### **BTVN 3**

**Mô tả**: Xuất các số chẵn từ mảng sử dụng 2 phương pháp

**Cho mảng**: `x = [[1,2,3,4,5,6,7,8,9,10]]`

**Kết quả**:
```
2
4
6
8
10
[np.int64(2), np.int64(4), np.int64(6), np.int64(8), np.int64(10)]
```

### **BTVN 4**

**Mô tả**: Xử lý dữ liệu sinh viên, tách train/test và tạo tập dữ liệu cho cross-validation

**Cách hoạt động**:
- Sử dụng `np.random.rand()` để tạo dữ liệu mẫu
- Slicing để tách cột: `data[:, :4]` và `data[:, 4]`
- Tính số lượng train: `int(0.7 * len(x))`
- Sử dụng `np.array_split()` để chia thành 10 tập bằng nhau

**Kết quả**:
```
X shape: (150, 4)
y shape: (150,)
Kích thước X: (150, 4)
Kích thước y: (150,)
Kích thước X_train: (105, 4), X_test: (45, 4)
Số lượng tập dữ liệu: 10
Kích thước mỗi tập: (15, 5)
```
