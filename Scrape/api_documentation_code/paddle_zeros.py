import paddle

data = paddle.zeros(shape=[3, 2], dtype='float32')
# [[0. 0.]
#  [0. 0.]
#  [0. 0.]]
data = paddle.zeros(shape=[2, 2])
# [[0. 0.]
#  [0. 0.]]

# shape is a Tensor
shape = paddle.full(shape=[2], dtype='int32', fill_value=2)
data3 = paddle.zeros(shape=shape, dtype='int32')
# [[0 0]
#  [0 0]]