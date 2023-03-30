import paddle.fluid as fluid
import paddle
paddle.enable_static()
# x is a Tensor variable with following elements:
#    [[0.2, 0.3, 0.5, 0.9]
#     [0.1, 0.2, 0.6, 0.7]]
# Each example is followed by the corresponding output tensor.
x = fluid.data(name='x', shape=[2, 4], dtype='float32')
fluid.layers.reduce_prod(x)  # [0.0002268]
fluid.layers.reduce_prod(x, dim=0)  # [0.02, 0.06, 0.3, 0.63]
fluid.layers.reduce_prod(x, dim=-1)  # [0.027, 0.0084]
fluid.layers.reduce_prod(x, dim=1,
                         keep_dim=True)  # [[0.027], [0.0084]]

# y is a Tensor variable with shape [2, 2, 2] and elements as below:
#      [[[1.0, 2.0], [3.0, 4.0]],
#      [[5.0, 6.0], [7.0, 8.0]]]
# Each example is followed by the corresponding output tensor.
y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
fluid.layers.reduce_prod(y, dim=[1, 2]) # [24.0, 1680.0]
fluid.layers.reduce_prod(y, dim=[0, 1]) # [105.0, 384.0]