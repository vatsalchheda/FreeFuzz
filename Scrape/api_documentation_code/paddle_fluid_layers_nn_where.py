import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy as np

# condition is a tensor [True, False, True]
condition = layers.assign(np.array([1, 0, 1], dtype='int32'))
condition = layers.cast(condition, 'bool')
out = layers.where(condition) # [[0], [2]]

# condition is a tensor [[True, False], [False, True]]
condition = layers.assign(np.array([[1, 0], [0, 1]], dtype='int32'))
condition = layers.cast(condition, 'bool')
out = layers.where(condition) # [[0, 0], [1, 1]]

# condition is a tensor [False, False, False]
condition = layers.assign(np.array([0, 0, 0], dtype='int32'))
condition = layers.cast(condition, 'bool')
out = layers.where(condition) # [[]]