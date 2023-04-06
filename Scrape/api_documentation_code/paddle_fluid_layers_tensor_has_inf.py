import paddle
data = paddle.randn(shape=[4, 32, 32], dtype="float32")
res = paddle.fluid.layers.has_inf(data)
# [False]