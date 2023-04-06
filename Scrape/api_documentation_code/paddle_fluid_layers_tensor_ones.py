import paddle.fluid as fluid
data0 = fluid.layers.ones(shape=[2, 4], dtype='float32') # [[1., 1., 1., 1.], [1., 1., 1., 1.]]

# shape is a Tensor
shape = fluid.layers.fill_constant(shape=[2], dtype='int32', value=2)
data1 = fluid.layers.ones(shape=shape, dtype='int32') #[[1, 1], [1, 1]]