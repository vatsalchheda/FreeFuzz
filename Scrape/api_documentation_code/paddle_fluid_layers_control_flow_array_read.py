# First we're going to create a LoDTensorArray, then we're going to write the Tensor into
# the specified position, and finally we're going to read the Tensor at that position.
import paddle.fluid as fluid
arr = fluid.layers.create_array(dtype='float32')
tmp = fluid.layers.fill_constant(shape=[3, 2], dtype='int64', value=5)
i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
# tmp is the Tensor with shape [3,2], and if we write it into the position with subscript 10
# of the empty-array: arr, then the length of arr becomes 11.
arr = fluid.layers.array_write(tmp, i, array=arr)
# Read the data of the position with subscript 10.
item = fluid.layers.array_read(arr, i)

# You can print out the data via executor.
input = fluid.layers.Print(item, message="The LoDTensor of the i-th position:")
main_program = fluid.default_main_program()
exe = fluid.Executor(fluid.CPUPlace())
exe.run(main_program)

# The printed result is:

# 1569588169  The LoDTensor of the i-th position: The place is:CPUPlace
# Tensor[array_read_0.tmp_0]
#    shape: [3,2,]
#    dtype: l
#    data: 5,5,5,5,5,5,

# the output is 2-D Tensor with shape [3,2].
# dtype is the corresponding C++ data type, which may vary in different environments.
# Eg: if the data type of tensor is int64, then the corresponding C++ data type is int64_t,
#       so the dtype value is typeid(int64_t).Name(), which is 'x' on MacOS, 'l' on Linux,
#       and '__int64' on Windows. They both represent 64-bit integer variables.