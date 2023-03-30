import paddle

x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
out1 = paddle.var(x)
# [2.66666667]
out2 = paddle.var(x, axis=1)
# [1.         4.33333333]