import numpy as np
import paddle.fluid as fluid

with fluid.dygraph.guard():
    inp = np.ones([3, 1024], dtype='float32')
    t = fluid.dygraph.base.to_variable(inp)
    linear1 = fluid.Linear(1024, 4, bias_attr=False)
    linear2 = fluid.Linear(4, 4)
    ret = linear1(t)
    dy_ret = linear2(ret)