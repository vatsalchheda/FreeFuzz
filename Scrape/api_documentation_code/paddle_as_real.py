import paddle
x = paddle.arange(12, dtype=paddle.float32).reshape([2, 3, 2])
y = paddle.as_complex(x)
z = paddle.as_real(y)
print(z)

# Tensor(shape=[2, 3, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[0. , 1. ],
#          [2. , 3. ],
#          [4. , 5. ]],

#         [[6. , 7. ],
#          [8. , 9. ],
#          [10., 11.]]])