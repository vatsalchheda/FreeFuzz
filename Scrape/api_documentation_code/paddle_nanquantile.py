import paddle

x = paddle.to_tensor(
    [[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]],
    dtype="float32")
x[0,0] = float("nan")

y1 = paddle.nanquantile(x, q=0.5, axis=[0, 1])
# Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
#        5.)

y2 = paddle.nanquantile(x, q=0.5, axis=1)
# Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=True,
#        [2.50000000, 7.        ])

y3 = paddle.nanquantile(x, q=[0.3, 0.5], axis=0)
# Tensor(shape=[2, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
#        [[5.        , 2.50000000, 3.50000000, 4.50000000, 5.50000000],
#         [5.        , 3.50000000, 4.50000000, 5.50000000, 6.50000000]])

y4 = paddle.nanquantile(x, q=0.8, axis=1, keepdim=True)
# Tensor(shape=[2, 1], dtype=float64, place=Place(cpu), stop_gradient=True,
#        [[3.40000000],
#         [8.20000000]])

nan = paddle.full(shape=[2, 3], fill_value=float("nan"))
y5 = paddle.nanquantile(nan, q=0.8, axis=1, keepdim=True)
# Tensor(shape=[2, 1], dtype=float64, place=Place(cpu), stop_gradient=True,
#        [[nan],
#         [nan]])