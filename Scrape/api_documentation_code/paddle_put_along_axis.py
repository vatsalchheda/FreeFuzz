import paddle

x = paddle.to_tensor([[10, 30, 20], [60, 40, 50]])
index = paddle.to_tensor([[0]])
value = 99
axis = 0
result = paddle.put_along_axis(x, index, value, axis)
print(result)
# [[99, 99, 99],
# [60, 40, 50]]