import paddle


def func(x, y):
    return paddle.matmul(x, y)


x = paddle.to_tensor([[1., 2.], [3., 4.]])
J = paddle.incubate.autograd.Jacobian(func, [x, x])
print(J[:, :])
# Tensor(shape=[4, 8], dtype=float32, place=Place(gpu:0), stop_gradient=False,
#        [[1., 3., 0., 0., 1., 0., 2., 0.],
#         [2., 4., 0., 0., 0., 1., 0., 2.],
#         [0., 0., 1., 3., 3., 0., 4., 0.],
#         [0., 0., 2., 4., 0., 3., 0., 4.]])

print(J[0, :])
# Tensor(shape=[8], dtype=float32, place=Place(gpu:0), stop_gradient=False,
#        [1., 3., 0., 0., 1., 0., 2., 0.])
print(J[:, 0])
# Tensor(shape=[4], dtype=float32, place=Place(gpu:0), stop_gradient=False,
#        [1., 2., 0., 0.])