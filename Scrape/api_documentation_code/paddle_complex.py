import paddle
x = paddle.arange(2, dtype=paddle.float32).unsqueeze(-1)
y = paddle.arange(3, dtype=paddle.float32)
z = paddle.complex(x, y)
print(z)
# Tensor(shape=[2, 3], dtype=complex64, place=Place(cpu), stop_gradient=True,
#        [[0j    , 1j    , 2j    ],
#         [(1+0j), (1+1j), (1+2j)]])