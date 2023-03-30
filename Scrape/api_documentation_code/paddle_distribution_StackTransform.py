import paddle

x = paddle.stack(
    (paddle.to_tensor([1., 2., 3.]), paddle.to_tensor([1, 2., 3.])), 1)
t = paddle.distribution.StackTransform(
    (paddle.distribution.ExpTransform(),
    paddle.distribution.PowerTransform(paddle.to_tensor(2.))),
    1
)
print(t.forward(x))
# Tensor(shape=[3, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[2.71828175 , 1.         ],
#         [7.38905621 , 4.         ],
#         [20.08553696, 9.         ]])

print(t.inverse(t.forward(x)))
# Tensor(shape=[3, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[1., 1.],
#         [2., 2.],
#         [3., 3.]])

print(t.forward_log_det_jacobian(x))
# Tensor(shape=[3, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[1.        , 0.69314718],
#         [2.        , 1.38629436],
#         [3.        , 1.79175949]])