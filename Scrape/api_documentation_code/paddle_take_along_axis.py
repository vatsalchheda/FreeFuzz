import paddle

x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7,8,9]])
index = paddle.to_tensor([[0]])
axis = 0
result = paddle.take_along_axis(x, index, axis)
print(result)
# [[1, 2, 3]]