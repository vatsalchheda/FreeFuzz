import paddle.fluid as fluid
x = fluid.layers.data(name="x", shape=[-1, 4])
out = fluid.contrib.layers.shuffle_batch(x)