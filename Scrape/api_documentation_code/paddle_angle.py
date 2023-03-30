import paddle

x = paddle.to_tensor([-2, -1, 0, 1]).unsqueeze(-1).astype('float32')
y = paddle.to_tensor([-2, -1, 0, 1]).astype('float32')
z = x + 1j * y
print(z)
# Tensor(shape=[4, 4], dtype=complex64, place=Place(cpu), stop_gradient=True,
#        [[(-2-2j), (-2-1j), (-2+0j), (-2+1j)],
#         [(-1-2j), (-1-1j), (-1+0j), (-1+1j)],
#         [-2j    , -1j    ,  0j    ,  1j    ],
#         [ (1-2j),  (1-1j),  (1+0j),  (1+1j)]])

theta = paddle.angle(z)
print(theta)
# Tensor(shape=[4, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[-2.35619450, -2.67794514,  3.14159274,  2.67794514],
#         [-2.03444386, -2.35619450,  3.14159274,  2.35619450],
#         [-1.57079637, -1.57079637,  0.        ,  1.57079637],
#         [-1.10714877, -0.78539819,  0.        ,  0.78539819]])