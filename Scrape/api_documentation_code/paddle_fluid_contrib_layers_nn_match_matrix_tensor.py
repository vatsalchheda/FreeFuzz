import numpy as np
from paddle.fluid import layers
from paddle.fluid import contrib

x_lod_tensor = layers.data(name='x', shape=[10], lod_level=1)
y_lod_tensor = layers.data(name='y', shape=[10], lod_level=1)
out, out_tmp = contrib.match_matrix_tensor(
    x=x_lod_tensor, y=y_lod_tensor, channel_num=3)