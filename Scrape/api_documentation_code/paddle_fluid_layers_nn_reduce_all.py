import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy as np

# x is a bool Tensor variable with following elements:
#    [[True, False]
#     [True, True]]
x = fluid.layers.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
x = fluid.layers.cast(x, 'bool')

out = fluid.layers.reduce_all(x)  # False
out = fluid.layers.reduce_all(x, dim=0)  # [True, False]
out = fluid.layers.reduce_all(x, dim=-1)  # [False, True]
# keep_dim=False, x.shape=(2,2), out.shape=(2,)

out = fluid.layers.reduce_all(x, dim=1, keep_dim=True)  # [[False], [True]]
# keep_dim=True, x.shape=(2,2), out.shape=(2,1)