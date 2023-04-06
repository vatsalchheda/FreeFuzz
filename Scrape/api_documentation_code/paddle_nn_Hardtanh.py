import paddle

x = paddle.to_tensor([-1.5, 0.3, 2.5])
m = paddle.nn.Hardtanh()
out = m(x) # [-1., 0.3, 1.]