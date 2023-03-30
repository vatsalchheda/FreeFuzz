import paddle

a = paddle.rand([3, 3], dtype="float32")
a_t = paddle.transpose(a, [1, 0])
x = paddle.matmul(a, a_t) + 1e-03

out = paddle.linalg.cholesky(x, upper=False)
print(out)