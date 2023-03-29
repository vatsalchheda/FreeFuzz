import paddle


x = paddle.to_tensor([1.,2.,3.])
t = paddle.distribution.StickBreakingTransform()
print(t.forward(x))
# Tensor(shape=[4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [0.47536686, 0.41287899, 0.10645414, 0.00530004])
print(t.inverse(t.forward(x)))
# Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [0.99999988, 2.        , 2.99999881])
print(t.forward_log_det_jacobian(x))
# Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [-9.10835075])