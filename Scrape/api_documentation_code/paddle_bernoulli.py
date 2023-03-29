import paddle

paddle.set_device('cpu')  # on CPU device
paddle.seed(100)

x = paddle.rand([2,3])
print(x)
# [[0.55355281, 0.20714243, 0.01162981],
#  [0.51577556, 0.36369765, 0.26091650]]

out = paddle.bernoulli(x)
print(out)
# [[1., 0., 1.],
#  [0., 1., 0.]]