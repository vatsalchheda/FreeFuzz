import paddle

x_int = paddle.arange(0, 12).reshape([3, 4])
x_float = x_int.astype(paddle.float64)

idx_pos = paddle.arange(4, 10).reshape([2, 3])  # positive index
idx_neg = paddle.arange(-2, 4).reshape([2, 3])  # negative index
idx_err = paddle.arange(-2, 13).reshape([3, 5])  # index out of range

paddle.take(x_int, idx_pos)
# Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[4, 5, 6],
#         [7, 8, 9]])

paddle.take(x_int, idx_neg)
# Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[10, 11, 0 ],
#         [1 , 2 , 3 ]])

paddle.take(x_float, idx_pos)
# Tensor(shape=[2, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
#        [[4., 5., 6.],
#         [7., 8., 9.]])

x_int.take(idx_pos)
# Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[4, 5, 6],
#         [7, 8, 9]])

paddle.take(x_int, idx_err, mode='wrap')
# Tensor(shape=[3, 5], dtype=int32, place=Place(cpu), stop_gradient=True,
#        [[10, 11, 0 , 1 , 2 ],
#         [3 , 4 , 5 , 6 , 7 ],
#         [8 , 9 , 10, 11, 0 ]])

paddle.take(x_int, idx_err, mode='clip')
# Tensor(shape=[3, 5], dtype=int32, place=Place(cpu), stop_gradient=True,
#        [[0 , 0 , 0 , 1 , 2 ],
#         [3 , 4 , 5 , 6 , 7 ],
#         [8 , 9 , 10, 11, 11]])