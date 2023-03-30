import paddle

tanh = paddle.distribution.TanhTransform()

x = paddle.to_tensor([[1., 2., 3.], [4., 5., 6.]])

print(tanh.forward(x))
# Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[0.76159418, 0.96402758, 0.99505478],
#         [0.99932933, 0.99990922, 0.99998772]])
print(tanh.inverse(tanh.forward(x)))
# Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[1.00000012, 2.        , 3.00000286],
#         [4.00002146, 5.00009823, 6.00039864]])
print(tanh.forward_log_det_jacobian(x))
# Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[-0.86756170 , -2.65000558 , -4.61865711 ],
#         [-6.61437654 , -8.61379623 , -10.61371803]])
print(tanh.inverse_log_det_jacobian(tanh.forward(x)))
# Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[0.86756176 , 2.65000558 , 4.61866283 ],
#         [6.61441946 , 8.61399269 , 10.61451530]])