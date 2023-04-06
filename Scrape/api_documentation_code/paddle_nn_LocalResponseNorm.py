import paddle

x = paddle.rand(shape=(3, 3, 112, 112), dtype="float32")
m = paddle.nn.LocalResponseNorm(size=5)
y = m(x)
print(y.shape)  # [3, 3, 112, 112]