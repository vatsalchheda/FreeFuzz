import paddle

# default, all axis will be used to exec fft
x = paddle.ones((2, 3, 4))
print(paddle.fft.rfftn(x))
# Tensor(shape=[2, 3, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
#        [[[(24+0j), 0j     , 0j     ],
#          [0j     , 0j     , 0j     ],
#          [0j     , 0j     , 0j     ]],
#
#         [[0j     , 0j     , 0j     ],
#          [0j     , 0j     , 0j     ],
#          [0j     , 0j     , 0j     ]]])

# use axes(2, 0)
print(paddle.fft.rfftn(x, axes=(2, 0)))
# Tensor(shape=[2, 3, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
#        [[[(8+0j), 0j     , 0j     ],
#          [(8+0j), 0j     , 0j     ],
#          [(8+0j), 0j     , 0j     ]],
#
#         [[0j     , 0j     , 0j     ],
#          [0j     , 0j     , 0j     ],
#          [0j     , 0j     , 0j     ]]])