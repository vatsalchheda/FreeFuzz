import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.floor(x)
print(out)
# [-1. -1.  0.  0.]