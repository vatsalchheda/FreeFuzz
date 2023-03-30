import paddle.fluid as fluid
data = fluid.layers.eye(3, dtype='int32')
# [[1, 0, 0]
#  [0, 1, 0]
#  [0, 0, 1]]

data = fluid.layers.eye(2, 3, dtype='int32')
# [[1, 0, 0]
#  [0, 1, 0]]

data = fluid.layers.eye(2, batch_shape=[3])
# Construct a batch of 3 identity tensors, each 2 x 2.
# data[i, :, :] is a 2 x 2 identity tensor, i = 0, 1, 2.