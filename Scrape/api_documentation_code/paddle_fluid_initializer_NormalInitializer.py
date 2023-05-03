import paddle
paddle.enable_static()
import paddle.fluid as fluid
x = fluid.data(name="data", shape=[None, 32, 32], dtype="float32")
fc = fluid.layers.fc(input=x, size=10,
    param_attr=fluid.initializer.Normal(loc=0.0, scale=2.0))