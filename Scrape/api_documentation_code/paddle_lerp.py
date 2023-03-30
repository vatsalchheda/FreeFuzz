import paddle

x = paddle.arange(1., 5., dtype='float32')
y = paddle.empty([4], dtype='float32')
y.fill_(10.)
out = paddle.lerp(x, y, 0.5)
# out: [5.5, 6., 6.5, 7.]