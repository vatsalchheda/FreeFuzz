import paddle

# default dtype for ones OP
data1 = paddle.ones(shape=[3, 2])
# [[1. 1.]
#  [1. 1.]
#  [1. 1.]]

data2 = paddle.ones(shape=[2, 2], dtype='int32')
# [[1 1]
#  [1 1]]

# shape is a Tensor
shape = paddle.full(shape=[2], dtype='int32', fill_value=2)
data3 = paddle.ones(shape=shape, dtype='int32')
# [[1 1]
#  [1 1]]