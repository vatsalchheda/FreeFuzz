import paddle

x = paddle.arange(1., 5., dtype='float32')
y = paddle.arange(1, 5, dtype='int32')
print(paddle.is_floating_point(x))
# True
print(paddle.is_floating_point(y))
# False