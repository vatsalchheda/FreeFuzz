import numpy as np
import paddle.fluid as fluid
x = fluid.layers.assign(np.array([2, 3, 3, 1, 5, 3], dtype='int32'))
out, index = fluid.layers.unique(x) # out is [2, 3, 1, 5]; index is [0, 1, 1, 2, 3, 1]