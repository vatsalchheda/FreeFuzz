import paddle
import paddle.fluid as fluid
paddle.enable_static()

x = fluid.data(name='x', shape=[-1, 5], dtype='float32')
index = fluid.data(name='index', shape=[-1, 1], dtype='int32')
output = fluid.layers.gather(x, index)