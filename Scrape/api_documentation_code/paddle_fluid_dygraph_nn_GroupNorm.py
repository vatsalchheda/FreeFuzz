import paddle.fluid as fluid
import numpy as np

with fluid.dygraph.guard():
    x = np.random.random((8, 32, 32)).astype('float32')
    groupNorm = fluid.dygraph.nn.GroupNorm(channels=32, groups=4)
    ret = groupNorm(fluid.dygraph.base.to_variable(x))