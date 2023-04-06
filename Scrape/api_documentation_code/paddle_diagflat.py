import paddle

x = paddle.to_tensor([[1, 2], [3, 4]])
y = paddle.diagflat(x)
print(y)
# Tensor(shape=[4, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[1, 0, 0, 0],
#         [0, 2, 0, 0],
#         [0, 0, 3, 0],
#         [0, 0, 0, 4]])

y = paddle.diagflat(x, offset=1)
print(y)
# Tensor(shape=[5, 5], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[0, 1, 0, 0, 0],
#         [0, 0, 2, 0, 0],
#         [0, 0, 0, 3, 0],
#         [0, 0, 0, 0, 4],
#         [0, 0, 0, 0, 0]])

y = paddle.diagflat(x, offset=-1)
print(y)
# Tensor(shape=[5, 5], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[0, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0],
#         [0, 2, 0, 0, 0],
#         [0, 0, 3, 0, 0],
#         [0, 0, 0, 4, 0]])