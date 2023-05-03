import paddle
paddle.enable_static()
import paddle.fluid as fluid
data = fluid.data(name="input", shape=[None, 784])
result = fluid.layers.softshrink(x=data, alpha=0.3)