# a square system of linear equations:
# x1 +   x2  +   x3 = 0
#      2*x2  +   x3 = -9
#               -x3 = 5

import paddle

x = paddle.to_tensor([[1, 1, 1],
                      [0, 2, 1],
                      [0, 0,-1]], dtype="float64")
y = paddle.to_tensor([[0], [-9], [5]], dtype="float64")
out = paddle.linalg.triangular_solve(x, y, upper=True)

print(out)
# [7, -2, -5]