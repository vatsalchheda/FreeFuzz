import paddle
import paddle.nn.functional as F

x = paddle.to_tensor([[-1, 0.5], [1, 1.5]])
out1 = F.gelu(x)
# [[-0.15865529,  0.34573123],
#  [ 0.84134471,  1.39978933]]
out2 = F.gelu(x, True)
# [[-0.15880799,  0.34571400],
#  [ 0.84119201,  1.39957154]]