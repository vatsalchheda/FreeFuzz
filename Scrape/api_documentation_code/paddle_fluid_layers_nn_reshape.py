import paddle
import paddle.fluid as fluid
paddle.enable_static()

# example 1:
# attr shape is a list which doesn't contain Tensors.
data_1 = fluid.data(
  name='data_1', shape=[2, 4, 6], dtype='float32')
reshaped_1 = fluid.layers.reshape(
  x=data_1, shape=[-1, 0, 3, 2])
# the shape of reshaped_1 is [2,4,3,2].

# example 2:
# attr shape is a list which contains Tensors.
data_2 = fluid.layers.fill_constant([2,25], "int32", 3)
dim = fluid.layers.fill_constant([1], "int32", 5)
reshaped_2 = fluid.layers.reshape(data_2, shape=[dim, 10])
# the shape of reshaped_2 is [5,10].

# example 3:
data_3 = fluid.data(
  name="data_3", shape=[2,4,6], dtype='float32')
reshaped_3 = fluid.layers.reshape(x=data_3, shape=[6,8])
# the shape of reshaped_3 is [6,8].