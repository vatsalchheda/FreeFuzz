import paddle

x = paddle.rand(shape=[4, 6], dtype='float32')
y = paddle.fluid.layers.isfinite(x)
print(y)