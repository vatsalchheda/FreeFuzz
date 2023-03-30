# a square system of linear equations:
# 2*X0 + X1 = 9
# X0 + 2*X1 = 8

import paddle

x = paddle.to_tensor([[3, 1],[1, 2]], dtype="float64")
y = paddle.to_tensor([9, 8], dtype="float64")
out = paddle.linalg.solve(x, y)

print(out)
# [2., 3.])