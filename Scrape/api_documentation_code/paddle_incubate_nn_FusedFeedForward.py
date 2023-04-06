# required: gpu
import paddle
from paddle.incubate.nn import FusedFeedForward

fused_feedforward_layer = FusedFeedForward(8, 8)
x = paddle.rand((1, 8, 8))
out = fused_feedforward_layer(x)
print(out.numpy().shape)
# (1, 8, 8)