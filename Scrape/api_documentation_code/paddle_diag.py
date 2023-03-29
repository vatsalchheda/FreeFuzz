import paddle

paddle.disable_static()
x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
y = paddle.diag(x)
print(y)
# Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [1, 5])

y = paddle.diag(x, offset=1)
print(y)
# Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [2, 6])

y = paddle.diag(x, offset=-1)
print(y)
# Tensor(shape=[1], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [4])