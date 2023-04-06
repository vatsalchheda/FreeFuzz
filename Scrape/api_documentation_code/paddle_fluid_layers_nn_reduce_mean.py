import paddle
import paddle.fluid as fluid
paddle.enable_static()

# x is a Tensor variable with following elements:
#    [[0.2, 0.3, 0.5, 0.9]
#     [0.1, 0.2, 0.6, 0.7]]
# Each example is followed by the corresponding output tensor.
x = fluid.data(name='x', shape=[2, 4], dtype='float32')
fluid.layers.reduce_mean(x)  # [0.4375]
fluid.layers.reduce_mean(x, dim=0)  # [0.15, 0.25, 0.55, 0.8]
fluid.layers.reduce_mean(x, dim=-1)  # [0.475, 0.4]
fluid.layers.reduce_mean(x, dim=1, keep_dim=True)  # [[0.475], [0.4]]

# y is a Tensor variable with shape [2, 2, 2] and elements as below:
#      [[[1.0, 2.0], [3.0, 4.0]],
#      [[5.0, 6.0], [7.0, 8.0]]]
# Each example is followed by the corresponding output tensor.
y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
fluid.layers.reduce_mean(y, dim=[1, 2]) # [2.5, 6.5]
fluid.layers.reduce_mean(y, dim=[0, 1]) # [4.0, 5.0]