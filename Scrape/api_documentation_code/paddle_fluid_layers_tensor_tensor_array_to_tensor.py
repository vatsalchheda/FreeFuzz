import paddle.fluid as fluid
import numpy as np
x0 = fluid.layers.assign(np.random.rand(2, 2).astype("float32"))
x1 = fluid.layers.assign(np.random.rand(2, 2).astype("float32"))
i = fluid.layers.fill_constant(shape=[1], dtype="int64", value=0)
array = fluid.layers.create_array(dtype='float32')
fluid.layers.array_write(x0, i, array)
fluid.layers.array_write(x1, i + 1, array)
output, output_index = fluid.layers.tensor_array_to_tensor(input=array)