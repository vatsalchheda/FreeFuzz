import paddle
x = paddle.to_tensor([-5, -1, 1])
res = paddle.bitwise_not(x)
print(res) # [4, 0, -2]