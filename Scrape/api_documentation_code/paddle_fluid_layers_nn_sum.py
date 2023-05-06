import paddle
paddle.enable_static()
import paddle.fluid as fluid

input0 = fluid.layers.fill_constant(shape=[2, 3], dtype='int64', value=5)
input1 = fluid.layers.fill_constant(shape=[2, 3], dtype='int64', value=3)
sum = fluid.layers.sum([input0, input1])

# You can print out 'sum' via executor.
out = fluid.layers.Print(sum, message="the sum of input0 and input1: ")
exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_main_program())

# The printed result is:
# 1570701754        the sum of input0 and input1:   The place is:CPUPlace
# Tensor[sum_0.tmp_0]
#    shape: [2,3,]
#    dtype: l
#    data: 8,8,8,8,8,8,

# the sum of input0 and input1 is 2-D Tensor with shape [2,3].
# dtype is the corresponding C++ data type, which may vary in different environments.
# Eg: if the data type of tensor is int64, then the corresponding C++ data type is int64_t,
#       so the dtype value is typeid(int64_t).Name(), which is 'x' on MacOS, 'l' on Linux,
#       and '__int64' on Windows. They both represent 64-bit integer variables.