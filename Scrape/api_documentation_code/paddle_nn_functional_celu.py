import paddle
import paddle.nn.functional as F
x = paddle.to_tensor([[-1., 6.], [1., 15.6]])
out = F.celu(x, alpha=0.2)
# [[-0.19865242,  6.        ],
#  [ 1.        , 15.60000038]]