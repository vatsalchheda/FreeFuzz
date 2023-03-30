import paddle.fluid as fluid

out1 = fluid.layers.range(0, 10, 2, 'int32')
# [0, 2, 4, 6, 8]

start_var = fluid.layers.fill_constant([1], 'int64', 3)
out2 = fluid.layers.range(start_var, 7, 1, 'int64')
# [3, 4, 5, 6]