import paddle.fluid as fluid
import paddle

paddle.enable_static()

data = fluid.data(name='data', shape=[None, 3, 32, 32, 32], dtype='float32')

# max pool3d
pool3d = fluid.layers.pool3d(
  input = data,
  pool_size = 2,
  pool_type = "max",
  pool_stride = 1,
  global_pooling=False)

# average pool3d
pool3d = fluid.layers.pool3d(
  input = data,
  pool_size = 2,
  pool_type = "avg",
  pool_stride = 1,
  global_pooling=False)

# global average pool3d
pool3d = fluid.layers.pool3d(
  input = data,
  pool_size = 2,
  pool_type = "avg",
  pool_stride = 1,
  global_pooling=True)

# example 1:
# Attr(pool_padding) is a list with 6 elements, Attr(data_format) is "NCDHW".
out_1 = fluid.layers.pool3d(
  input = data,
  pool_size = 2,
  pool_type = "avg",
  pool_stride = 1,
  pool_padding = [1, 2, 1, 0, 1, 2],
  global_pooling = False,
  data_format = "NCDHW")

# example 2:
# Attr(pool_padding) is a string, Attr(data_format) is "NCDHW".
out_2 = fluid.layers.pool3d(
  input = data,
  pool_size = 3,
  pool_type = "avg",
  pool_stride = 1,
  pool_padding = "VALID",
  global_pooling = False,
  data_format = "NCDHW")