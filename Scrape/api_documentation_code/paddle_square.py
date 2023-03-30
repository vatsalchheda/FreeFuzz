import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.square(x)
print(out)
# [0.16 0.04 0.01 0.09]