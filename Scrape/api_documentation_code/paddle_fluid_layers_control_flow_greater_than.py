import paddle.fluid as fluid
import numpy as np
label = fluid.layers.assign(np.array([2, 3], dtype='int32'))
limit = fluid.layers.assign(np.array([3, 2], dtype='int32'))
out = fluid.layers.greater_than(x=label, y=limit) #out=[False, True]
out1 = label > limit #out1=[False, True]