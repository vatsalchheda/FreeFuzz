import paddle
import paddle.nn.functional as F

label = paddle.randn((10,1))
prob = paddle.randn((10,1))
cost = F.log_loss(input=prob, label=label)