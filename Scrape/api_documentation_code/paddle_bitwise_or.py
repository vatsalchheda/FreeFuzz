import paddle
x = paddle.to_tensor([-5, -1, 1])
y = paddle.to_tensor([4,  2, -3])
res = paddle.bitwise_or(x, y)
print(res)  # [-1, -1, -3]