import paddle.fluid as fluid
import paddle

paddle.enable_static()

data = fluid.data(name='data', shape=[None, 3, 32, 32], dtype='float32')

# max pool2d
pool2d = fluid.layers.pool2d(
  input = data,
  pool_size = 2,
  pool_type = "max",
  pool_stride = 1,
  global_pooling=False)

# average pool2d
pool2d = fluid.layers.pool2d(
  input = data,
  pool_size = 2,
  pool_type = "avg",
  pool_stride = 1,
  global_pooling=False)

# global average pool2d
pool2d = fluid.layers.pool2d(
  input = data,
  pool_size = 2,
  pool_type = "avg",
  pool_stride = 1,
  global_pooling=True)

# Attr(pool_padding) is a list with 4 elements, Attr(data_format) is "NCHW".
out_1 = fluid.layers.pool2d(
  input = data,
  pool_size = 3,
  pool_type = "avg",
  pool_stride = 1,
  pool_padding = [1, 2, 1, 0],
  data_format = "NCHW")

# Attr(pool_padding) is a string, Attr(data_format) is "NCHW".
out_2 = fluid.layers.pool2d(
  input = data,
  pool_size = 3,
  pool_type = "avg",
  pool_stride = 1,
  pool_padding = "VALID",
  data_format = "NCHW")