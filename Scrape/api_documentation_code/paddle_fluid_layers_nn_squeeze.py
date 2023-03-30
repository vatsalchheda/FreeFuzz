import paddle.fluid as fluid
import paddle.fluid.layers as layers
# set batch size=None
x = fluid.data(name='x', shape=[None, 5, 1, 10])
y = layers.squeeze(input=x, axes=[2]) # y.shape=[None, 5, 10]