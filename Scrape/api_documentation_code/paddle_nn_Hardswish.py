import paddle

x = paddle.to_tensor([-4., 5., 1.])
m = paddle.nn.Hardswish()
out = m(x) # [0., 5., 0.666667]