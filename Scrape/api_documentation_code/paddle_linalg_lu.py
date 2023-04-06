import paddle

x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
lu,p,info = paddle.linalg.lu(x, get_infos=True)

# >>> lu:
# Tensor(shape=[3, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
#    [[5.        , 6.        ],
#        [0.20000000, 0.80000000],
#        [0.60000000, 0.50000000]])
# >>> p
# Tensor(shape=[2], dtype=int32, place=CUDAPlace(0), stop_gradient=True,
#    [3, 3])
# >>> info
# Tensor(shape=[], dtype=int32, place=CUDAPlace(0), stop_gradient=True,
#    0)

P,L,U = paddle.linalg.lu_unpack(lu,p)

# >>> P
# (Tensor(shape=[3, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
# [[0., 1., 0.],
# [0., 0., 1.],
# [1., 0., 0.]]),
# >>> L
# Tensor(shape=[3, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
# [[1.        , 0.        ],
# [0.20000000, 1.        ],
# [0.60000000, 0.50000000]]),
# >>> U
# Tensor(shape=[2, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
# [[5.        , 6.        ],
# [0.        , 0.80000000]]))


# one can verify : X = P @ L @ U ;