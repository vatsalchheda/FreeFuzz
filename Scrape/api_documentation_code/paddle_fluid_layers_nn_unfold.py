import paddle
import paddle.nn.functional as F

x = paddle.randn((100,3,224,224))
y = F.unfold(x, [3, 3], 1, 1, 1)