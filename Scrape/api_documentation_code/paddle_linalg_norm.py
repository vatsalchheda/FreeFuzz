import paddle
x = paddle.arange(24, dtype="float32").reshape([2, 3, 4]) - 12
# x: Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
#          [[[-12., -11., -10., -9. ],
#            [-8. , -7. , -6. , -5. ],
#            [-4. , -3. , -2. , -1. ]],

#           [[ 0. ,  1. ,  2. ,  3. ],
#            [ 4. ,  5. ,  6. ,  7. ],
#            [ 8. ,  9. ,  10.,  11.]]])

# compute frobenius norm along last two dimensions.
out_fro = paddle.linalg.norm(x, p='fro', axis=[0,1])
# out_fro: Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
#                 [17.43559647, 16.91153526, 16.73320007, 16.91153526])

# compute 2-order vector norm along last dimension.
out_pnorm = paddle.linalg.norm(x, p=2, axis=-1)
# out_pnorm: Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
#                [[21.11871147, 13.19090557, 5.47722578 ],
#                 [3.74165750 , 11.22497177, 19.13112640]])

# compute 2-order  norm along [0,1] dimension.
out_pnorm = paddle.linalg.norm(x, p=2, axis=[0,1])
# out_pnorm: Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
#                  [17.43559647, 16.91153526, 16.73320007, 16.91153526])

# compute inf-order  norm
out_pnorm = paddle.linalg.norm(x, p=float("inf"))
# out_pnorm  = Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
#                    [12.])

out_pnorm = paddle.linalg.norm(x, p=float("inf"), axis=0)
# out_pnorm: Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
#                 [[12., 11., 10., 9. ],
#                  [8. , 7. , 6. , 7. ],
#                  [8. , 9. , 10., 11.]])

# compute -inf-order  norm
out_pnorm = paddle.linalg.norm(x, p=-float("inf"))
# out_pnorm: Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
#                  [0.])

out_pnorm = paddle.linalg.norm(x, p=-float("inf"), axis=0)
# out_pnorm: Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
#                  [[0., 1., 2., 3.],
#                  [4., 5., 6., 5.],
#                  [4., 3., 2., 1.]])