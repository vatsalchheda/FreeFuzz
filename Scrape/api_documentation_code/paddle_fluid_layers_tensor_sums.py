import paddle.fluid as fluid

x0 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=1)
x1 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=2)
x2 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=3)
x3 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=0)

# Sum of multiple Tensors, the result is stored to a new Variable sum0 (sum0=x0+x1+x2, the value is [[6, ..., 6], ..., [6, ..., 6]])
sum0 = fluid.layers.sums(input=[x0, x1, x2])

# Sum of multiple Tensors, sum1 and x3 represents the same Variable (x3=x0+x1+x2, the value is [[6, ..., 6], ..., [6, ..., 6]])
sum1 = fluid.layers.sums(input=[x0, x1, x2], out=x3)