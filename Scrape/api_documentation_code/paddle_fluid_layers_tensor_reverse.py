import paddle.fluid as fluid
import numpy as np
data = fluid.layers.assign(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype='float32')) # [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]
result1 = fluid.layers.reverse(data, 0) # [[6., 7., 8.], [3., 4., 5.], [0., 1., 2.]]
result2 = fluid.layers.reverse(data, [0, 1]) # [[8., 7., 6.], [5., 4., 3.], [2., 1., 0.]]

# example of LoDTensorArray
data1 = fluid.layers.assign(np.array([[0, 1, 2]], dtype='float32'))
data2 = fluid.layers.assign(np.array([[3, 4, 5]], dtype='float32'))
tensor_array = fluid.layers.create_array(dtype='float32')
i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
fluid.layers.array_write(data1, i, tensor_array)
fluid.layers.array_write(data2, i+1, tensor_array)

reversed_tensor_array = fluid.layers.reverse(tensor_array, 0) # {[[3, 4, 5]], [[0, 1, 2]]}