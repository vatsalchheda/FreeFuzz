import paddle
import paddle.fluid as fluid
paddle.enable_static()

x = fluid.data(name='x', shape=[3, 4, 5], dtype='float32')
index = fluid.data(name='index', shape=[2, 2], dtype='int32')
output = fluid.layers.gather_nd(x, index)