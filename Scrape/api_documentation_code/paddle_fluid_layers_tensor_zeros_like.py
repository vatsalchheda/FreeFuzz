import paddle
paddle.enable_static()
import paddle.fluid as fluid
x = fluid.data(name='x', dtype='float32', shape=[3])
data = fluid.layers.zeros_like(x) # [0.0, 0.0, 0.0]