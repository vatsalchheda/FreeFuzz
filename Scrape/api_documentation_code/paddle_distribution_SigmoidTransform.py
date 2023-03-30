import paddle

x = paddle.ones((2,3))
t = paddle.distribution.SigmoidTransform()
print(t.forward(x))
# Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[0.73105860, 0.73105860, 0.73105860],
#         [0.73105860, 0.73105860, 0.73105860]])
print(t.inverse(t.forward(x)))
# Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[1.00000012, 1.00000012, 1.00000012],
#         [1.00000012, 1.00000012, 1.00000012]])
print(t.forward_log_det_jacobian(x))
# Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[-1.62652326, -1.62652326, -1.62652326],
#         [-1.62652326, -1.62652326, -1.62652326]])