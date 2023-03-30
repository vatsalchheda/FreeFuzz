import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.cosh(x)
print(out)
# [1.08107237 1.02006676 1.00500417 1.04533851]