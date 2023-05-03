import paddle
paddle.enable_static()
import paddle.fluid as fluid
input = fluid.data(
    name='data', shape=[1], dtype='float32')
reward = fluid.layers.clip(x=input, min=-1.0, max=1.0)