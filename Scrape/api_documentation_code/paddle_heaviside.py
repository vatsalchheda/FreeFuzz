import paddle
x = paddle.to_tensor([-0.5, 0, 0.5])
y = paddle.to_tensor([0.1])
paddle.heaviside(x, y)
#    [0.        , 0.10000000, 1.        ]
x = paddle.to_tensor([[-0.5, 0, 0.5], [-0.5, 0.5, 0]])
y = paddle.to_tensor([0.1, 0.2, 0.3])
paddle.heaviside(x, y)
#    [[0.        , 0.20000000, 1.        ],
#     [0.        , 1.        , 0.30000001]]