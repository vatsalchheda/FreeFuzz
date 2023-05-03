import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import numpy as np
import paddle

# x's shape is [1, 3, 1, 2]
x = np.array([[[[1.0, 8.0]], [[10.0, 5.0]], [[4.0, 6.0]]]]).astype('float32')
with fluid.dygraph.guard():
    x = to_variable(x)
    instanceNorm = paddle.fluid.dygraph.nn.InstanceNorm(3)
    ret = instanceNorm(x)
    # ret's shape is [1, 3, 1, 2]; value is [-1 1 0.999999 -0.999999 -0.999995 0.999995]
    print(ret)