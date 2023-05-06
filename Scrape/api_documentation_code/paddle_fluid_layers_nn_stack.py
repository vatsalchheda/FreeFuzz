import paddle
paddle.enable_static()
import paddle.fluid as fluid
import paddle.fluid.layers as layers
# set batch size=None
x1 = fluid.data(name='x1', shape=[None, 1, 2], dtype='int32')
x2 = fluid.data(name='x2', shape=[None, 1, 2], dtype='int32')
# stack Tensor list
data = layers.stack([x1,x2]) # stack according to axis 0, data.shape=[2, None, 1, 2]

data = layers.stack([x1,x2], axis=1) # stack according to axis 1, data.shape=[None, 2, 1, 2]