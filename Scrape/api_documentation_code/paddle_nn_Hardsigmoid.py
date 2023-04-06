import paddle

m = paddle.nn.Hardsigmoid()
x = paddle.to_tensor([-4., 5., 1.])
out = m(x) # [0., 1, 0.666667]