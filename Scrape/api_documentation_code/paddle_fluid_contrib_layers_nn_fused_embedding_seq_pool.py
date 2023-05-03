import paddle
paddle.enable_static()
import numpy as np
import paddle.fluid as fluid

dict_size = 20
data_t = fluid.layers.data(
    name='word', shape=[1], dtype='int64', lod_level=1)
padding_idx = np.random.randint(1, 10)
out = fluid.contrib.fused_embedding_seq_pool(
    input=data_t,
    size=[dict_size, 32],
    param_attr='w',
    padding_idx=padding_idx,
    is_sparse=False)