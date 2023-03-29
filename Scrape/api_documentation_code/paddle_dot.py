import paddle

# 1-D Tensor * 1-D Tensor
x = paddle.to_tensor([1, 2, 3])
y = paddle.to_tensor([4, 5, 6])
z = paddle.dot(x, y)
print(z)  # [32]

# 2-D Tensor * 2-D Tensor
x = paddle.to_tensor([[1, 2, 3], [2, 4, 6]])
y = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
z = paddle.dot(x, y)
print(z)  # [[32], [64]]