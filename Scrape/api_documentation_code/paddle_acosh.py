import paddle

x = paddle.to_tensor([1., 3., 4., 5.])
out = paddle.acosh(x)
print(out)
# [0.        , 1.76274729, 2.06343699, 2.29243159]