import numpy as np
from paddle.fluid import layers
from paddle.fluid import contrib

x_lod_tensor = layers.data(name='x', shape=[1], lod_level=1)
row_lod_tensor = layers.data(name='row', shape=[6], lod_level=1)
col_lod_tensor = layers.data(name='col', shape=[6], lod_level=1)
out = contrib.sequence_topk_avg_pooling(input=x_lod_tensor,
                                       row=row_lod_tensor,
                                       col=col_lod_tensor,
                                       topks=[1, 3, 5],
                                       channel_num=5)