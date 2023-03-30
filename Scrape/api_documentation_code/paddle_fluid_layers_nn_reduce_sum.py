import paddle.fluid as fluid
import paddle
paddle.enable_static()
# x is a Tensor variable with following elements:
#    [[0.2, 0.3, 0.5, 0.9]
#     [0.1, 0.2, 0.6, 0.7]]
# Each example is followed by the corresponding output tensor.
x = fluid.data(name='x', shape=[2, 4], dtype='float32')
fluid.layers.reduce_sum(x)  # [3.5]
fluid.layers.reduce_sum(x, dim=0)  # [0.3, 0.5, 1.1, 1.6]
fluid.layers.reduce_sum(x, dim=-1)  # [1.9, 1.6]
fluid.layers.reduce_sum(x, dim=1, keep_dim=True)  # [[1.9], [1.6]]

# y is a Tensor variable with shape [2, 2, 2] and elements as below:
#      [[[1, 2], [3, 4]],
#      [[5, 6], [7, 8]]]
# Each example is followed by the corresponding output tensor.
y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
fluid.layers.reduce_sum(y, dim=[1, 2]) # [10, 26]
fluid.layers.reduce_sum(y, dim=[0, 1]) # [16, 20]