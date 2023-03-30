import paddle
data = paddle.randn(shape=[2,3], dtype="float32")
res = paddle.fluid.layers.has_nan(data)
# [False]