import paddle

data = paddle.arange(1, 13, dtype="int64").reshape([3,-1])
# Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[1 , 2 , 3 , 4 ],
#         [5 , 6 , 7 , 8 ],
#         [9 , 10, 11, 12]])

tril1 = paddle.tril(data)
# Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[1 , 0 , 0 , 0 ],
#         [5 , 6 , 0 , 0 ],
#         [9 , 10, 11, 0 ]])

# example 2, positive diagonal value
tril2 = paddle.tril(data, diagonal=2)
# Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[1 , 2 , 3 , 0 ],
#         [5 , 6 , 7 , 8 ],
#         [9 , 10, 11, 12]])

# example 3, negative diagonal value
tril3 = paddle.tril(data, diagonal=-1)
# Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[0 , 0 , 0 , 0 ],
#         [5 , 0 , 0 , 0 ],
#         [9 , 10, 0 , 0 ]])