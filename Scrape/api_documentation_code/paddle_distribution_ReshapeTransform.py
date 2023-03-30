import paddle

x = paddle.ones((1,2,3))
reshape_transform = paddle.distribution.ReshapeTransform((2, 3), (3, 2))
print(reshape_transform.forward_shape((1,2,3)))
# (5, 2, 6)
print(reshape_transform.forward(x))
# Tensor(shape=[1, 3, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[1., 1.],
#          [1., 1.],
#          [1., 1.]]])
print(reshape_transform.inverse(reshape_transform.forward(x)))
# Tensor(shape=[1, 2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[1., 1., 1.],
#          [1., 1., 1.]]])
print(reshape_transform.forward_log_det_jacobian(x))
# Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [0.])