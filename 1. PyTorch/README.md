# DEEP LEARNING

- **Sinh Viên Thực Hiên:** Phạm Thế Hùng
- **MSSV:** 2374802010164
- **Môn Học:** Giới Thiệu Học Sâu
- **Giảng viên:** Nguyễn Thái Anh

## PyTorch Basic

## Công nghệ  sử dụng

- **Python**: ngôn ngữ dùng để code chính
- **PyTorch (`torch`)**: tensor, autograd (tính đạo hàm), thao tác với CUDA/GPU
- **NumPy (`numpy`)**: tạo mảng và minh hoạ việc chia sẻ bộ nhớ khi convert sang tensor
- **Pandas (`pandas`)**: đọc file CSV và xem dữ liệu Iris
- **Matplotlib (`matplotlib`)**: có import (chưa dùng nhiều trong notebook)
- **scikit-learn (`sklearn`)**: chia tập train/test, encode nhãn
---

### Kiểm tra GPU / CUDA

- Dùng `torch.cuda.is_available()` để kiểm tra xem máy có GPU và xài được CUDA hay không

### Dataset với PyTorch (Iris.csv)

- Dùng `pandas.read_csv` để đọc file Iris.csv, dữ liệu có 150 dòng và 6 cột.

### Tính đạo hàm bằng PyTorch

- Tạo x với requires_grad=True định nghĩa x = 2, rồi dùng backward()` để PyTorch tự tính đạo hàm.
- Với ví dụ \(y = 2x^4 + x^3 + 3x^2 + 5x + 1\) tại x=2 thì y = 63 và x.grad = 93

### BTVN 01:
- Bắt đầu với x = 1, mỗi vòng tính (y = 5x^5 + 6x^3 - 3x + 1), sau đó cập nhật x = x - alpha * x.grad với `alpha = 0.1`.
- Thì cho ra kết quả = 9 và độ dốc là 40

### BTVN 02
- Bắt đầu với `x = 2`, mỗi vòng tính (y = x^3 + 2x^2 + 5x + 1), sau đó cập nhật `x = x - alpha * x.grad` với `alpha = 0.1`.
- ví dụ như vòng lặp đầu tiên thì x = 2 và đạo hàm da thức trên và thay vào ct ta định nghĩa x = x - alpha * x.grad ta được
    -  2 - 0.1 * 3(2)^2 + 4(2) + 5 = -0.5 cứ thế lặp lại 9 lần còn lại
 
### BTVN 03

- Tạo dữ liệu giả: `x` là số giờ tự học, `y` được sinh theo công thức gần đúng (y = 3x + 5 + noise).
- Khởi tạo ngẫu nhiên `w`, `b`, tính loss MSE giữa `y_pred = w*x + b` và `y`, rồi dùng Gradient Descent để cập nhật `w`, `b` nhiều vòng.
- Sau nhiều vòng lặp, loss giảm dần và `w`, `b` tiến gần về 3 và 5 (do dữ liệu sinh ra từ công thức đó).
- Kết quả w sẽ tiến gần 3 và b tiến gần 5, loss giảm theo thời gian

### Python with Tensor

### Kiểm tra phiển bản pytorch
```python
torch.__version__
```
kết quả trả về '2.9.1+cpu'
```python
#Chuyển đổi mảng numpy sang tensor pytorch
arr = np.array([1,2,3,4,5])
print(arr)
print(arr.dtype)
print(type(arr))
```
kết quả là in ra mảng [1 2 3 4 5] với kiểu dữ liệu số nguyên và kiểu mảng numpy

```python
arr = np.array([1, 2, 3, 4, 5])
x = torch.from_numpy(arr)

print(x)
print(x.dtype)
print(x.type())
```
- giờ là từ dạng mảng numpy chuyển qua dạng mảng tỏch

Mảng 2 chiều

```python
arr2 = np.arange(0, 12).reshape(4, 3)
x2 = torch.from_numpy(arr2)

print(x2)
print(x2.type())
```
kết quả là in ra mảng 2 chiều với kiểu dữ liệu số thực của pytorch

### BTVN 04
- Trường hợp 1: torch.from_numpy() tạo tensor dùng chung dữ liệu với NumPy array nên khi thay đổi NumPy array thì tensor cũng thay đổi theo
- Trường hopwj 2: torch.tensor() tạo tensor bằng cách sao chép dữ liệu nên tensor không bị ảnh hưởng khi NumPy array thay đổi

### BTVN 05
- `torch.empty()`: tạo tensor “rác” chưa khởi tạo giá trị
- `torch.zeros()`: toàn 0
- `torch.ones()`: toàn 1
- `torch.rand()`: random số ngẫu nhiên từ 0 đến 1
- `view()`  `reshape()`: đổi shape miễn tổng số phần tử không đổi

