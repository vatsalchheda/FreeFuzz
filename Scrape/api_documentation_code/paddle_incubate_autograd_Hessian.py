import paddle


def reducer(x):
    return paddle.sum(x * x)


x = paddle.rand([2, 2])
h = paddle.incubate.autograd.Hessian(reducer, x)
print(h[:])
# Tensor(shape=[4, 4], dtype=float32, place=Place(gpu:0), stop_gradient=False,
#        [[2., 0., 0., 0.],
#         [0., 2., 0., 0.],
#         [0., 0., 2., 0.],
#         [0., 0., 0., 2.]])