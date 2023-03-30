import paddle

data1 = paddle.full(shape=[2,1], fill_value=0, dtype='int64')
#[[0]
# [0]]

# attr shape is a list which contains Tensor.
positive_2 = paddle.full([1], 2, "int32")
data3 = paddle.full(shape=[1, positive_2], dtype='float32', fill_value=1.5)
# [[1.5 1.5]]

# attr shape is a Tensor.
shape = paddle.full([2], 2, "int32")
data4 = paddle.full(shape=shape, dtype='bool', fill_value=True)
# [[True True]
#  [True True]]

# attr fill_value is a Tensor.
val = paddle.full([1], 2.0, "float32")
data5 = paddle.full(shape=[2,1], fill_value=val, dtype='float32')
# [[2.0]
#  [2.0]]