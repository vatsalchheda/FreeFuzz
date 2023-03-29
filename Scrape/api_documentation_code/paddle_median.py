import paddle

x = paddle.arange(12).reshape([3, 4])
# Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[0 , 1 , 2 , 3 ],
#         [4 , 5 , 6 , 7 ],
#         [8 , 9 , 10, 11]])

y1 = paddle.median(x)
# Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [5.50000000])

y2 = paddle.median(x, axis=0)
# Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [4., 5., 6., 7.])

y3 = paddle.median(x, axis=1)
# Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [1.50000000, 5.50000000, 9.50000000])

y4 = paddle.median(x, axis=0, keepdim=True)
# Tensor(shape=[1, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[4., 5., 6., 7.]])