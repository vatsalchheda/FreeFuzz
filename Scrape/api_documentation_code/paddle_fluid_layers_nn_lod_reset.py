import paddle.fluid as fluid
x = fluid.layers.data(name='x', shape=[10])
y = fluid.layers.data(name='y', shape=[10, 20], lod_level=2)
out = fluid.layers.lod_reset(x=x, y=y)