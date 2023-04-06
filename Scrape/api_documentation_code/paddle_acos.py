import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.acos(x)
print(out)
# [1.98231317 1.77215425 1.47062891 1.26610367]