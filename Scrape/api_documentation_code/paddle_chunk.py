import paddle

x = paddle.rand([3, 9, 5])

out0, out1, out2 = paddle.chunk(x, chunks=3, axis=1)
# out0.shape [3, 3, 5]
# out1.shape [3, 3, 5]
# out2.shape [3, 3, 5]


# axis is negative, the real axis is (rank(x) + axis) which real
# value is 1.
out0, out1, out2 = paddle.chunk(x, chunks=3, axis=-2)
# out0.shape [3, 3, 5]
# out1.shape [3, 3, 5]
# out2.shape [3, 3, 5]