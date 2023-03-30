import paddle
import paddle.fluid as fluid
paddle.enable_static()
input = fluid.data(name='input', shape=[None,4,2,2], dtype='float32')
out = fluid.layers.shuffle_channel(x=input, group=2)