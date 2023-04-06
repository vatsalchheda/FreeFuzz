import paddle.fluid as fluid
data = fluid.layers.zeros(shape=[3, 2], dtype='float32') # [[0., 0.], [0., 0.], [0., 0.]]

# shape is a Tensor
shape = fluid.layers.fill_constant(shape=[2], dtype='int32', value=2)
data1 = fluid.layers.zeros(shape=shape, dtype='int32') #[[0, 0], [0, 0]]