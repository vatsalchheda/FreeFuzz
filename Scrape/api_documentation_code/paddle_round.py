import paddle

x = paddle.to_tensor([-0.5, -0.2, 0.6, 1.5])
out = paddle.round(x)
print(out)
# [-1. -0.  1.  2.]