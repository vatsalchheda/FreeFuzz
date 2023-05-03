import numpy as np
import paddle.fluid as fluid
x = fluid.layers.assign(np.array([2, 3, 3, 1, 5, 3], dtype='int32'))
out, index, count = fluid.layers.unique_with_counts(x) # out is [2, 3, 1, 5]; index is [0, 1, 1, 2, 3, 1]
                                          # count is [1, 3, 1, 1]
# x.shape=(6,) out.shape=(4,), index.shape=(6,), count.shape=(4,)