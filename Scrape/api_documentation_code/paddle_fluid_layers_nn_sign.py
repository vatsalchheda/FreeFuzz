import paddle.fluid as fluid
import numpy as np

# [1.0, 0.0, -1.0]
data = fluid.layers.sign(np.array([3.0, 0.0, -2.0], dtype='float32'))