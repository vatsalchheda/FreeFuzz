import paddle
data = paddle.logspace(0, 10, 5, 2, 'float32')
# [1.          , 5.65685415  , 32.         , 181.01933289, 1024.       ]
data = paddle.logspace(0, 10, 1, 2, 'float32')
# [1.]