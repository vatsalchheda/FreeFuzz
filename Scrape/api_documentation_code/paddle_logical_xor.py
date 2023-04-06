import paddle

x = paddle.to_tensor([True, False], dtype="bool").reshape([2, 1])
y = paddle.to_tensor([True, False, True, False], dtype="bool").reshape([2, 2])
res = paddle.logical_xor(x, y)
print(res)
# Tensor(shape=[2, 2], dtype=bool, place=Place(cpu), stop_gradient=True,
#        [[False, True ],
#         [True , False]])