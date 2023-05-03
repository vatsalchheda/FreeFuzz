import paddle
paddle.enable_static()
import paddle.fluid as fluid
data = fluid.layers.data(name="input", shape=[32, 784])
result = fluid.layers.cumsum(data, axis=0)