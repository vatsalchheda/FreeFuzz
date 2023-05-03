import paddle
paddle.enable_static()
# The following code completes the function: subtract 10 from the data greater than 0 in x, add 10 to the data less than 0 in x, and sum all the data.
import numpy as np
import paddle.fluid as fluid

x = fluid.layers.data(name='x', shape=[4, 1], dtype='float32', append_batch_size=False)
y = fluid.layers.data(name='y', shape=[4, 1], dtype='float32', append_batch_size=False)

x_d = np.array([[3], [1], [-2], [-3]]).astype(np.float32)
y_d = np.zeros((4, 1)).astype(np.float32)

# Compare the size of x, y pairs of elements, output cond, cond is shape [4, 1], data type bool 2-D tensor.
# Based on the input data x_d, y_d, it can be inferred that the data in cond are [[true], [true], [false], [false]].
cond = fluid.layers.greater_than(x, y)
# Unlike other common OPs, ie below returned by the OP is an IfElse OP object
ie = fluid.layers.IfElse(cond)

with ie.true_block():
    # In this block, according to cond condition, the data corresponding to true dimension in X is obtained and subtracted by 10.
    out_1 = ie.input(x)
    out_1 = out_1 - 10
    ie.output(out_1)
with ie.false_block():
    # In this block, according to cond condition, get the data of the corresponding condition in X as false dimension, and add 10
    out_1 = ie.input(x)
    out_1 = out_1 + 10
    ie.output(out_1)

# According to cond condition, the data processed in the two blocks are merged. The output here is output, the type is List, and the element type in List is Variable.
output = ie() #  [array([[-7.], [-9.], [ 8.], [ 7.]], dtype=float32)]

# Get the first Variable in the output List and add all elements.
out = fluid.layers.reduce_sum(output[0])

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

res = exe.run(fluid.default_main_program(), feed={"x":x_d, "y":y_d}, fetch_list=[out])
print(res)
# [array([-1.], dtype=float32)]