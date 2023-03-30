import paddle

x = paddle.to_tensor([-1, 0.3, 2.5])
m = paddle.nn.Hardshrink()
out = m(x) # [-1., 0., 2.5]