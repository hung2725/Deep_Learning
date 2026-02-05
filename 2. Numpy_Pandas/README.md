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
---
# Pandas

## Công nghệ sử dụng
- **Python 3**
- **Pandas**
- **NumPy**

### Series

- **Series** = mảng một chiều có chỉ mục (index).
- Tạo từ **list** (vd: `pd.Series([0.5, 0.3, 1.0, 2.5])`).
- Tạo từ **NumPy array** (vd: `np.arange(5)`).
- **Attributes**: `.values` (mảng), `.index` (chỉ mục).
- **Indexing**: truy cập theo chỉ số (vd: `data_pd[1]`), slice (vd: `data_pd[-2:]`). Lưu ý với index mặc định không dùng được `-1` như list.
- **Letter indexing**: gán index bằng chữ (vd: `index=['a','b','c','d']`), truy cập bằng `data_pd['b']`.
- Có thể kết hợp index chữ và số.

### DataFrame

- **DataFrame** = cấu trúc 2 chiều: hàng có index, cột có tên; bản chất là nhiều Series cùng index.
- Tạo từ **dictionary** key = tên cột, value = list/array của cột.
- Truy cập cột: kiểu dictionary `df['tên_cột']` hoặc kiểu thuộc tính `df.tên_cột` không dùng được khi tên cột trùng tên, ví dụ như `pop`.

### iloc vs. loc

- **loc**: truy cập theo **nhãn index** (label); slice bao gồm cả điểm cuối (vd: `loc[1:3]` gồm 3).
- **iloc**: truy cập theo **vị trí số** (position); slice không gồm điểm cuối (vd: `iloc[1:3]` là vị trí 1, 2).
- Nên dùng `iloc` khi muốn slice theo vị trí để tránh nhầm với index nguyên.

### DataFrame indexing / slicing / fancy indexing

- **Dictionary style**: `df['cột']`, `df[['cột1','cột2']]`.
- **Attribute style**: `df.cột` (hạn chế khi tên cột đặc biệt hoặc trùng method).
- **Feature engineering**: thêm cột mới bằng gán (vd: `df['cột_mới'] = ...`).
- **iloc và loc**: chọn hàng/cột theo vị trí hoặc nhãn; kết hợp dạng `df.loc[:, 'cột']`, `df.iloc[0:5, 1:4]`.
- **Masking + fancy**: dùng boolean array để lọc hàng.
- **Indexing vs. slicing**: quy ước quan trọng (label vs position, bao gồm/không bao gồm điểm cuối).

### Broadcasting

- **Basic broadcasting**: phép toán giữa DataFrame/Series với scalar hoặc giữa Series với DataFrame theo cột/hàng.
- **Index alignment**: pandas tự căn theo index khi cộng/trừ/nhân/chia; chỗ không khớp thành NaN.
- **DataFrame và Series**: khi tính toán, Series được broadcast theo hàng hoặc cột tùy trục.

### Bài tập về nhà 1

**Cách hoạt động:** Đọc file `howlongwelive.csv` vào DataFrame, xem nhanh dữ liệu (2 dòng đầu/cuối, shape, tên cột, bảng `.describe()`). Sau đó làm sạch: xóa hai cột **Hepatitis B** và **Population** vì nhiều NaN (Hepatitis B còn tương quan cao với Diphtheria). Cột **Status** (Developing/Developed) được chuyển sang số 0/1 bằng `.map()`. Đổi tên cột thinness cho thống nhất. Cuối cùng tách dữ liệu: mọi cột trừ **Life expectancy** chuyển sang NumPy thành **X**, cột **Life expectancy** chuyển sang NumPy thành **y** 

**Kết quả:** DataFrame 2938 × 20 (sau khi xóa 2 cột). **X** là mảng đặc trưng, **y** là mảng nhãn tuổi thọ.


### Handling missing data

- **np.nan**: giá trị thiếu trong Pandas/NumPy.
- **isnull()**: kiểm tra ô nào là NaN.
- **not null**: kiểm tra ô không phải NaN.
- **dropna()**: xóa hàng/cột chứa NaN.
- **fillna()**: thay NaN bằng giá trị (vd: mean, median, 0). Lưu ý fillna có thể làm giảm phương sai; notebook có thảo luận nên bỏ qua hay thay thế.

### DataFrame concatenation & Join

- **concat**: nối DataFrame theo trục (axis=0: chồng hàng; axis=1: thêm cột). Mặc định join='outer'; có thể trùng index.
- **Join inner**: chỉ giữ phần index trùng nhau khi nối.

### Merging Datasets with ID

- Gộp hai bảng theo cột chung (ID) bằng **merge** (vd: kiểu SQL: inner, left, right, outer).
- Dùng khi có hai nguồn dữ liệu cần ghép theo khóa.

### Aggregation

- Các hàm gộp: **mean, median, min, max, std, var, sum**, v.v. áp dụng trên cột hoặc theo nhóm (sau groupby).
- Ví dụ: `df.mean()`, `df.groupby('cột').mean()`.

### Bài tập về nhà 2

**Cách hoạt động:** Làm việc với DataFrame (đã xử lý ở Bài 1 hoặc đọc lại CSV). Đầu tiên kiểm tra số giá trị thiếu từng cột rồi điền toàn bộ NaN bằng **trung bình (mean)** của từng cột tương ứng. Dùng **groupby theo Country** để tính tuổi thọ trung bình mỗi nước, từ đó tìm nước có tuổi thọ thấp nhất và cao nhất. **Groupby theo Status** để so sánh tuổi thọ trung bình giữa nước phát triển (Developed) và đang phát triển (Developing). Tiếp theo tạo một DataFrame phụ gồm cột **ID** (trùng với Country) và cột **Noise_level** (giá trị ngẫu nhiên), rồi **merge** với DataFrame chính theo cột ID để ghép thông tin vào.

**Kết quả:** Bảng không còn NaN; biết quốc gia tuổi thọ thấp và cao nhất và sự chênh lệch giữa Developed và Developing (Developed cao hơn rõ rệt); bảng cuối có thêm cột Noise_level sau merge.
