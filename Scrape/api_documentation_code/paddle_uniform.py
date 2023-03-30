import paddle

# example 1:
# attr shape is a list which doesn't contain Tensor.
out1 = paddle.uniform(shape=[3, 4])
# [[ 0.84524226,  0.6921872,   0.56528175,  0.71690357], # random
#  [-0.34646994, -0.45116323, -0.09902662, -0.11397249], # random
#  [ 0.433519,    0.39483607, -0.8660099,   0.83664286]] # random

# example 2:
# attr shape is a list which contains Tensor.
dim1 = paddle.to_tensor([2], 'int64')
dim2 = paddle.to_tensor([3], 'int32')
out2 = paddle.uniform(shape=[dim1, dim2])
# [[-0.9951253,   0.30757582, 0.9899647 ], # random
#  [ 0.5864527,   0.6607096,  -0.8886161]] # random

# example 3:
# attr shape is a Tensor, the data type must be int64 or int32.
shape_tensor = paddle.to_tensor([2, 3])
out3 = paddle.uniform(shape_tensor)
# [[-0.8517412,  -0.4006908,   0.2551912 ], # random
#  [ 0.3364414,   0.36278176, -0.16085452]] # random