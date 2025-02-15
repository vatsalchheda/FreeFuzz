import paddle

# example 1: attr shape is a list which doesn't contain Tensor.
out1 = paddle.standard_normal(shape=[2, 3])
# [[-2.923464  ,  0.11934398, -0.51249987],  # random
#  [ 0.39632758,  0.08177969,  0.2692008 ]]  # random

# example 2: attr shape is a list which contains Tensor.
dim1 = paddle.to_tensor([2], 'int64')
dim2 = paddle.to_tensor([3], 'int32')
out2 = paddle.standard_normal(shape=[dim1, dim2, 2])
# [[[-2.8852394 , -0.25898588],  # random
#   [-0.47420555,  0.17683524],  # random
#   [-0.7989969 ,  0.00754541]],  # random
#  [[ 0.85201347,  0.32320443],  # random
#   [ 1.1399018 ,  0.48336947],  # random
#   [ 0.8086993 ,  0.6868893 ]]]  # random

# example 3: attr shape is a Tensor, the data type must be int64 or int32.
shape_tensor = paddle.to_tensor([2, 3])
out3 = paddle.standard_normal(shape_tensor)
# [[-2.878077 ,  0.17099959,  0.05111201]  # random
#  [-0.3761474, -1.044801  ,  1.1870178 ]]  # random