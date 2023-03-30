import paddle.fluid as fluid

# example 1:
data_1 = fluid.layers.fill_constant(shape=[2, 3, 1], dtype='int32', value=0)
expanded_1 = fluid.layers.expand(data_1, expand_times=[1, 2, 2])
# the shape of expanded_1 is [2, 6, 2].

# example 2:
data_2 = fluid.layers.fill_constant(shape=[12, 14], dtype="int32", value=3)
expand_times = fluid.layers.fill_constant(shape=[2], dtype="int32", value=4)
expanded_2 = fluid.layers.expand(data_2, expand_times=expand_times)
# the shape of expanded_2 is [48, 56].