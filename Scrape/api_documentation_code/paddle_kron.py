import paddle
x = paddle.to_tensor([[1, 2], [3, 4]], dtype='int64')
y = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='int64')
out = paddle.kron(x, y)
print(out)
#        [[1, 2, 3, 2, 4, 6],
#         [ 4,  5,  6,  8, 10, 12],
#         [ 7,  8,  9, 14, 16, 18],
#         [ 3,  6,  9,  4,  8, 12],
#         [12, 15, 18, 16, 20, 24],
#         [21, 24, 27, 28, 32, 36]])