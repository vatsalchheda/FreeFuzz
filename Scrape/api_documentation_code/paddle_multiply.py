import paddle

x = paddle.to_tensor([[1, 2], [3, 4]])
y = paddle.to_tensor([[5, 6], [7, 8]])
res = paddle.multiply(x, y)
print(res) # [[5, 12], [21, 32]]

x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
y = paddle.to_tensor([2])
res = paddle.multiply(x, y)
print(res) # [[[2, 4, 6], [2, 4, 6]]]