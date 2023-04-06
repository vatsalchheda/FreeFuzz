# required: gpu
import paddle
from paddle.incubate.nn import FusedLinear

x = paddle.randn([3, 4])
linear = FusedLinear(4, 5)
y = linear(x)
print(y.shape) # [3, 5]