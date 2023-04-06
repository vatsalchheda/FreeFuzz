import paddle.fluid as fluid

x = fluid.layers.data(name='x', dtype='float32', shape=[3], append_batch_size=False)
data = fluid.layers.ones_like(x) # [1.0, 1.0, 1.0]