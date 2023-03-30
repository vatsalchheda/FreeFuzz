import paddle

xt = paddle.rand((3,4))
paddle.linalg.cov(xt)

'''
Tensor(shape=[3, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
    [[0.07918842, 0.06127326, 0.01493049],
        [0.06127326, 0.06166256, 0.00302668],
        [0.01493049, 0.00302668, 0.01632146]])
'''