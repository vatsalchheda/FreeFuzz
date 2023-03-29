import paddle
x1 = paddle.to_tensor([180.0, -180.0, 360.0, -360.0, 90.0, -90.0])
result1 = paddle.deg2rad(x1)
print(result1)
# Tensor(shape=[6], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#         [3.14159274, -3.14159274,  6.28318548, -6.28318548,  1.57079637,
#           -1.57079637])

x2 = paddle.to_tensor(180)
result2 = paddle.deg2rad(x2)
print(result2)
# Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#         [3.14159274])