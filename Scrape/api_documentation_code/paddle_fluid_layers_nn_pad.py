# x is a rank 2 tensor variable
import paddle.fluid as fluid
x = fluid.data(name='data', shape=[300, 300], dtype='float32')
out = fluid.layers.pad(x=x, paddings=[0, 1, 1, 2], pad_value=0.)