import paddle.fluid as fluid
import paddle
paddle.enable_static()

input = fluid.data(
    name='data',
    shape=[None, 256, 32, 32],
    dtype='float32')
out = fluid.layers.maxout(input, groups=2)