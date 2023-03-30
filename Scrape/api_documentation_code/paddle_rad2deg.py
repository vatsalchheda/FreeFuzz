import paddle
import math

x1 = paddle.to_tensor([3.142, -3.142, 6.283, -6.283, 1.570, -1.570])
result1 = paddle.rad2deg(x1)
print(result1)
# Tensor(shape=[6], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#         [180.02334595, -180.02334595,  359.98937988, -359.98937988,
#           9.95437622 , -89.95437622])

x2 = paddle.to_tensor(math.pi/2)
result2 = paddle.rad2deg(x2)
print(result2)
# Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#         [90.])

x3 = paddle.to_tensor(1)
result3 = paddle.rad2deg(x3)
print(result3)
# Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#         [57.29578018])