import paddle

x = paddle.to_tensor([2, 3, 4], 'float64')
y = paddle.to_tensor([1, 5, 2], 'float64')
z = paddle.add(x, y)
print(z)  # [3., 8., 6. ]