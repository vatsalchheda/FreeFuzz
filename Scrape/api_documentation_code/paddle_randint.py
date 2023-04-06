import paddle

# example 1:
# attr shape is a list which doesn't contain Tensor.
out1 = paddle.randint(low=-5, high=5, shape=[3])
# [0, -3, 2]  # random

# example 2:
# attr shape is a list which contains Tensor.
dim1 = paddle.to_tensor([2], 'int64')
dim2 = paddle.to_tensor([3], 'int32')
out2 = paddle.randint(low=-5, high=5, shape=[dim1, dim2])
# [[0, -1, -3],  # random
#  [4, -2,  0]]  # random

# example 3:
# attr shape is a Tensor
shape_tensor = paddle.to_tensor(3)
out3 = paddle.randint(low=-5, high=5, shape=shape_tensor)
# [-2, 2, 3]  # random

# example 4:
# data type is int32
out4 = paddle.randint(low=-5, high=5, shape=[3], dtype='int32')
# [-5, 4, -4]  # random

# example 5:
# Input only one parameter
# low=0, high=10, shape=[1], dtype='int64'
out5 = paddle.randint(10)
# [7]  # random