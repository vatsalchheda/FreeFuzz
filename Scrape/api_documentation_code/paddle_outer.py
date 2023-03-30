import paddle
x = paddle.arange(1, 4).astype('float32')
y = paddle.arange(1, 6).astype('float32')
out = paddle.outer(x, y)
print(out)
#        ([[1, 2, 3, 4, 5],
#         [2, 4, 6, 8, 10],
#         [3, 6, 9, 12, 15]])