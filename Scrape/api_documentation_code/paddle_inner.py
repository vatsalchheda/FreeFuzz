import paddle
x = paddle.arange(1, 7).reshape((2, 3)).astype('float32')
y = paddle.arange(1, 10).reshape((3, 3)).astype('float32')
out = paddle.inner(x, y)
print(out)
#        ([[14, 32, 50],
#         [32, 77, 122]])