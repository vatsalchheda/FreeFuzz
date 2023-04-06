import paddle

x = paddle.to_tensor([-5., 0., 5.])
m = paddle.nn.Mish()
out = m(x) # [-0.03357624, 0., 4.99955208]