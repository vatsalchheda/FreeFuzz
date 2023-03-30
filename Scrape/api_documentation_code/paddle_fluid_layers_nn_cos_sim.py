import paddle

x = paddle.rand(shape=[3, 7], dtype='float32')
y = paddle.rand(shape=[1, 7], dtype='float32')
out = paddle.fluid.layers.cos_sim(x, y)
print(out)