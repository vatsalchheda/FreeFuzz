import paddle
paddle.set_device('cpu')
paddle.seed(100)

x = paddle.uniform([2,3], min=1.0, max=5.0)
out = paddle.poisson(x)
#[[2., 5., 0.],
# [5., 1., 3.]]