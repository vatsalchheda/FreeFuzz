import paddle.fluid as fluid
import numpy as np
label = fluid.layers.assign(np.array([1, 3], dtype='int32'))
limit = fluid.layers.assign(np.array([1, 2], dtype='int32'))
out = fluid.layers.less_equal(x=label, y=limit) #out=[True, False]
out1 = label<= limit #out1=[True, False]