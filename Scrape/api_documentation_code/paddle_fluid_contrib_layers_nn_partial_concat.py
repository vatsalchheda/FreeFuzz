import paddle
paddle.enable_static()
import paddle.fluid as fluid
x = fluid.data(name="x", shape=[None,3], dtype="float32")
y = fluid.data(name="y", shape=[None,3], dtype="float32")
concat = fluid.contrib.layers.partial_concat(
    [x, y], start_index=0, length=2)