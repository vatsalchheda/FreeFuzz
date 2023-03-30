import paddle

u = paddle.to_tensor([[1, 1, 1],
                        [0, 2, 1],
                        [0, 0,-1]], dtype="float64")
b = paddle.to_tensor([[0], [-9], [5]], dtype="float64")
out = paddle.linalg.cholesky_solve(b, u, upper=True)

print(out)
# [-2.5, -7, 9.5]