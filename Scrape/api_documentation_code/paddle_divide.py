import paddle

x = paddle.to_tensor([2, 3, 4], dtype='float64')
y = paddle.to_tensor([1, 5, 2], dtype='float64')
z = paddle.divide(x, y)
print(z)  # [2., 0.6, 2.]