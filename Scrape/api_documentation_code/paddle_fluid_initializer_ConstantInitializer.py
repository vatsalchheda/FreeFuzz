import paddle
import paddle.fluid as fluid
paddle.enable_static()
x = fluid.data(name="data", shape=[8, 32, 32], dtype="float32")
fc = fluid.layers.fc(
    input=x,
    size=10,
    param_attr=fluid.initializer.Constant(value=2.0))