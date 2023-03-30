import paddle

y = paddle.arange(0, 8 ,dtype="float32").reshape([4, 2])
# Tensor(shape=[4, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[0., 1.],
#         [2., 3.],
#         [4., 5.],
#         [6., 7.]])

y1 = paddle.quantile(y, q=0.5, axis=[0, 1])
# Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
#        3.50000000)

y2 = paddle.quantile(y, q=0.5, axis=1)
# Tensor(shape=[4], dtype=float64, place=Place(cpu), stop_gradient=True,
#        [0.50000000, 2.50000000, 4.50000000, 6.50000000])

y3 = paddle.quantile(y, q=[0.3, 0.5], axis=0)
# Tensor(shape=[2, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
#        [[1.80000000, 2.80000000],
#         [3.        , 4.        ]])

y[0,0] = float("nan")
y4 = paddle.quantile(y, q=0.8, axis=1, keepdim=True)
# Tensor(shape=[4, 1], dtype=float64, place=Place(cpu), stop_gradient=True,
#        [[nan       ],
#         [2.80000000],
#         [4.80000000],
#         [6.80000000]])