import paddle
x = paddle.to_tensor([[float('nan'), 2. , 3. ], [0. , 1. , 2. ]])

y1 = x.nanmedian()
# y1 is [[2.]]

y2 = x.nanmedian(0)
# y2 is [[0.,  1.5, 2.5]]

y3 = x.nanmedian(0, keepdim=False)
# y3 is [0.,  1.5, 2.5]

y4 = x.nanmedian((0, 1))
# y4 is [[2.]]