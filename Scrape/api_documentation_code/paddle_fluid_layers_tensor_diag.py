# [[3, 0, 0]
#  [0, 4, 0]
#  [0, 0, 5]

import paddle.fluid as fluid
import numpy as np
diagonal = np.arange(3, 6, dtype='int32')
data = fluid.layers.diag(diagonal)
# diagonal.shape=(3,) data.shape=(3, 3)