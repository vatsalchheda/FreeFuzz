import paddle
x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.erf(x)
print(out)
# [-0.42839236 -0.22270259  0.11246292  0.32862676]