import paddle


def func(x):
    return paddle.matmul(x, x)


x = paddle.ones(shape=[2, 2], dtype='float32')
_, jvp_result = paddle.incubate.autograd.jvp(func, x)
print(jvp_result)
# Tensor(shape=[2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=False,
#        [[4., 4.],
#         [4., 4.]])
v = paddle.to_tensor([[1.0, 0.0], [0.0, 0.0]])
_, jvp_result = paddle.incubate.autograd.jvp(func, x, v)
print(jvp_result)
# Tensor(shape=[2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=False,
#        [[2., 1.],
#         [1., 0.]])