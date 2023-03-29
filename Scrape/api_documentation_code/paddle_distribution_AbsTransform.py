import paddle

abs = paddle.distribution.AbsTransform()

print(abs.forward(paddle.to_tensor([-1., 0., 1.])))
# Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [1., 0., 1.])

print(abs.inverse(paddle.to_tensor(1.)))
# (Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [-1.]), Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [1.]))

# The |dX/dY| is constant 1. So Log|dX/dY| == 0
print(abs.inverse_log_det_jacobian(paddle.to_tensor(1.)))
# (Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        0.), Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        0.))

#Special case handling of 0.
print(abs.inverse(paddle.to_tensor(0.)))
# (Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [0.]), Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [0.]))
print(abs.inverse_log_det_jacobian(paddle.to_tensor(0.)))
# (Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        0.), Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        0.))