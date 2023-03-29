import paddle.fluid as fluid

x = fluid.data(name='x', shape=[None, 3, 101, 60], dtype='float32')
guide = fluid.data(name='guide', shape=[None, 101, 60], dtype='float32')
grid = fluid.data(name='grid', shape=[None, 12, 8, 10, 6], dtype='float32')

# without offset
output = fluid.contrib.bilateral_slice(x, guide, grid, has_offset=False)

# has offset
output = fluid.contrib.bilateral_slice(x, guide, grid, has_offset=True)