import paddle

exp = paddle.distribution.ExpTransform()
print(exp.forward(paddle.to_tensor([1., 2., 3.])))
# Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [2.71828175 , 7.38905621 , 20.08553696])

print(exp.inverse(paddle.to_tensor([1., 2., 3.])))
# Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [0.        , 0.69314718, 1.09861231])

print(exp.forward_log_det_jacobian(paddle.to_tensor([1., 2., 3.])))
# Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [1., 2., 3.])

print(exp.inverse_log_det_jacobian(paddle.to_tensor([1., 2., 3.])))
# Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [ 0.        , -0.69314718, -1.09861231])