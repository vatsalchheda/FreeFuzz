System Message: ERROR/3 (/usr/local/lib/python3.8/site-packages/paddle/fluid/contrib/layers/nn.py:docstring of paddle.fluid.contrib.layers.nn.fused_embedding_seq_pool, line 34)
Error in “code-block” directive: maximum 1 argument(s) allowed, 9 supplied.
.. code-block:: python
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