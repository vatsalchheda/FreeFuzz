# x: [M, N], vec: [N]
# paddle.mv(x, vec)  # out: [M]

import paddle

x = paddle.to_tensor([[2, 1, 3], [3, 0, 1]]).astype("float64")
vec = paddle.to_tensor([3, 5, 1]).astype("float64")
out = paddle.mv(x, vec)
print(out)
# Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=True,
#        [14., 10.])