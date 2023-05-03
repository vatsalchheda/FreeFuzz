import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import numpy as np

with fluid.dygraph.guard():
   data = np.random.random((3, 32, 32, 5)).astype('float32')
   pool2d = fluid.dygraph.Pool2D(pool_size=2,
                  pool_type='max',
                  pool_stride=1,
                  global_pooling=False)
   pool2d_res = pool2d(to_variable(data))