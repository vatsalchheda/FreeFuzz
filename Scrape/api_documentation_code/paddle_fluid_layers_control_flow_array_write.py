import paddle.fluid as fluid
tmp = fluid.layers.fill_constant(shape=[3, 2], dtype='int64', value=5)
i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
# Write tmp into the position of arr with subscript 10 and return arr.
arr = fluid.layers.array_write(tmp, i=i)

# Now, arr is a LoDTensorArray with length 11. We can use array_read OP to read
# the data at subscript 10 and print it out.
item = fluid.layers.array_read(arr, i=i)
input = fluid.layers.Print(item, message="The content of i-th LoDTensor:")
main_program = fluid.default_main_program()
exe = fluid.Executor(fluid.CPUPlace())
exe.run(main_program)

# The printed result is:
# 1570533133    The content of i-th LoDTensor:  The place is:CPUPlace
# Tensor[array_read_0.tmp_0]
#    shape: [3,2,]
#    dtype: l
#    data: 5,5,5,5,5,5,

# the output is 2-D Tensor with shape [3,2], which is tmp above.
# dtype is the corresponding C++ data type, which may vary in different environments.
# Eg: if the data type of tensor is int64, then the corresponding C++ data type is int64_t,
#       so the dtype value is typeid(int64_t).Name(), which is 'x' on MacOS, 'l' on Linux,
#       and '__int64' on Windows. They both represent 64-bit integer variables.