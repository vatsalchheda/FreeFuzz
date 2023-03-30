import paddle

x = paddle.to_tensor([[-1. ,6.], [1., 15.6]])
m = paddle.nn.CELU(0.2)
out = m(x)
# [[-0.19865242,  6.        ],
#  [ 1.        , 15.60000038]]