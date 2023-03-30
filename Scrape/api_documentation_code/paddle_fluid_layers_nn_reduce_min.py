import paddle.fluid as fluid
import paddle
paddle.enable_static()

# x is a Tensor variable with following elements:
#    [[0.2, 0.3, 0.5, 0.9]
#     [0.1, 0.2, 0.6, 0.7]]
# Each example is followed by the corresponding output tensor.
x = fluid.data(name='x', shape=[2, 4], dtype='float32')
fluid.layers.reduce_min(x)  # [0.1]
fluid.layers.reduce_min(x, dim=0)  # [0.1, 0.2, 0.5, 0.7]
fluid.layers.reduce_min(x, dim=-1)  # [0.2, 0.1]
fluid.layers.reduce_min(x, dim=1, keep_dim=True)  # [[0.2], [0.1]]

# y is a Tensor variable with shape [2, 2, 2] and elements as below:
#      [[[1.0, 2.0], [3.0, 4.0]],
#      [[5.0, 6.0], [7.0, 8.0]]]
# Each example is followed by the corresponding output tensor.
y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
fluid.layers.reduce_min(y, dim=[1, 2]) # [1.0, 5.0]
fluid.layers.reduce_min(y, dim=[0, 1]) # [1.0, 2.0]