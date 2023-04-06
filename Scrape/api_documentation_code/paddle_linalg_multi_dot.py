import paddle
# A * B
A = paddle.rand([3, 4])
B = paddle.rand([4, 5])
out = paddle.linalg.multi_dot([A, B])
print(out.shape)
# [3, 5]
# A * B * C
A = paddle.rand([10, 5])
B = paddle.rand([5, 8])
C = paddle.rand([8, 7])
out = paddle.linalg.multi_dot([A, B, C])
print(out.shape)
# [10, 7]