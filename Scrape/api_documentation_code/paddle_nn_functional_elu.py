import paddle
import paddle.nn.functional as F

x = paddle.to_tensor([[-1., 6.], [1., 15.6]])
out = F.elu(x, alpha=0.2)
# [[-0.12642411  6.        ]
#  [ 1.          15.6      ]]