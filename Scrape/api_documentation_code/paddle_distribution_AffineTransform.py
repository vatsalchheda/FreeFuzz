import paddle

x = paddle.to_tensor([1., 2.])
affine = paddle.distribution.AffineTransform(paddle.to_tensor(0.), paddle.to_tensor(1.))

print(affine.forward(x))
# Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [1., 2.])
print(affine.inverse(affine.forward(x)))
# Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [1., 2.])
print(affine.forward_log_det_jacobian(x))
# Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [0.])