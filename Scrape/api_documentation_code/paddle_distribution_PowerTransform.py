import paddle

x = paddle.to_tensor([1., 2.])
power = paddle.distribution.PowerTransform(paddle.to_tensor(2.))

print(power.forward(x))
# Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [1., 4.])
print(power.inverse(power.forward(x)))
# Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [1., 2.])
print(power.forward_log_det_jacobian(x))
# Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [0.69314718, 1.38629436])