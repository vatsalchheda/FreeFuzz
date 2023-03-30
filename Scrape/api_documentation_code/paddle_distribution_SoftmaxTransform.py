import paddle

x = paddle.ones((2,3))
t = paddle.distribution.SoftmaxTransform()
print(t.forward(x))
# Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[0.33333334, 0.33333334, 0.33333334],
#         [0.33333334, 0.33333334, 0.33333334]])
print(t.inverse(t.forward(x)))
# Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[-1.09861231, -1.09861231, -1.09861231],
#         [-1.09861231, -1.09861231, -1.09861231]])