import paddle
import paddle.fluid as fluid
paddle.enable_static()

# example 1:
# attr shape is a list which doesn't contain Tensor.
result_1 = fluid.layers.uniform_random(shape=[3, 4])
# [[ 0.84524226,  0.6921872,   0.56528175,  0.71690357],
#  [-0.34646994, -0.45116323, -0.09902662, -0.11397249],
#  [ 0.433519,    0.39483607, -0.8660099,   0.83664286]]

# example 2:
# attr shape is a list which contains Tensor.
dim_1 = fluid.layers.fill_constant([1], "int64", 2)
dim_2 = fluid.layers.fill_constant([1], "int32", 3)
result_2 = fluid.layers.uniform_random(shape=[dim_1, dim_2])
# [[-0.9951253,   0.30757582, 0.9899647 ],
#  [ 0.5864527,   0.6607096,  -0.8886161 ]]

# example 3:
# attr shape is a Tensor, the data type must be int64 or int32.
var_shape = fluid.data(name='var_shape', shape=[2], dtype="int64")
result_3 = fluid.layers.uniform_random(var_shape)
# if var_shape's value is [2, 3]
# result_3 is:
# [[-0.8517412,  -0.4006908,   0.2551912 ],
#  [ 0.3364414,   0.36278176, -0.16085452]]