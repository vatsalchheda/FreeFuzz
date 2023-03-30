import paddle
x = paddle.arange(12, dtype=paddle.float32).reshape([2, 3, 2])
y = paddle.as_complex(x)
print(y)

# Tensor(shape=[2, 3], dtype=complex64, place=Place(gpu:0), stop_gradient=True,
#        [[1j      , (2+3j)  , (4+5j)  ],
#         [(6+7j)  , (8+9j)  , (10+11j)]])