import paddle
paddle.enable_static()
import paddle.fluid as fluid
data = fluid.layers.data(name="input", shape=[784])
result = fluid.layers.hard_shrink(x=data, threshold=0.3)