import paddle

multinomial = paddle.distribution.Multinomial(10, paddle.to_tensor([0.2, 0.3, 0.5]))
print(multinomial.sample((2, 3)))
# Tensor(shape=[2, 3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[1., 4., 5.],
#          [0., 2., 8.],
#          [2., 4., 4.]],

#         [[1., 6., 3.],
#          [3., 3., 4.],
#          [3., 4., 3.]]])