import paddle
paddle.enable_static()
import paddle.fluid as fluid
data = fluid.layers.data(name='x', shape=[784], dtype='float32')