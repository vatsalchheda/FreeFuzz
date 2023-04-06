import paddle

x = paddle.to_tensor([2, 3, 8, 7])
y = paddle.to_tensor([1, 5, 3, 3])
z = paddle.floor_divide(x, y)
print(z)  # [2, 0, 2, 2]