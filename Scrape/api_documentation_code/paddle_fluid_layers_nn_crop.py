import paddle.fluid as fluid
import paddle.fluid as fluid
import paddle
paddle.enable_static()
x = fluid.data(name="x", shape=[3, 3, 5], dtype="float32")
y = fluid.data(name="y", shape=[2, 2, 3], dtype="float32")
crop = fluid.layers.crop(x, shape=y)

# or
z = fluid.data(name="z", shape=[3, 3, 5], dtype="float32")
crop = fluid.layers.crop(z, shape=[2, 2, 3])