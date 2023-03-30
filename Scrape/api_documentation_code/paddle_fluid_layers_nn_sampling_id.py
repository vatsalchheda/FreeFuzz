import paddle.fluid as fluid
x = fluid.data(
    name="X",
    shape=[13, 11],
    dtype='float32')

out = fluid.layers.sampling_id(x)