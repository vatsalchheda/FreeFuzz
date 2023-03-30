import paddle.fluid as fluid
tmp = fluid.layers.zeros(shape=[10], dtype='int32')
i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
# tmp is 1-D Tensor with shape [10]. We write tmp into arr on subscript 10,
# then the length of arr becomes 11.
arr = fluid.layers.array_write(tmp, i=i)
# return the length of arr
arr_len = fluid.layers.array_length(arr)

# You can use executor to print out the length of LoDTensorArray.
input = fluid.layers.Print(arr_len, message="The length of LoDTensorArray:")
main_program = fluid.default_main_program()
exe = fluid.Executor(fluid.CPUPlace())
exe.run(main_program)

# The printed result is:

# 1569576542  The length of LoDTensorArray:   The place is:CPUPlace
# Tensor[array_length_0.tmp_0]
#    shape: [1,]
#    dtype: l
#    data: 11,

# 1-D Tensor with shape [1], whose value is 11. It means that the length of LoDTensorArray
# is 11.
# dtype is the corresponding C++ data type, which may vary in different environments.
# Eg: if the data type of tensor is int64, then the corresponding C++ data type is int64_t,
#       so the dtype value is typeid(int64_t).Name(), which is 'x' on MacOS, 'l' on Linux,
#       and '__int64' on Windows. They both represent 64-bit integer variables.