import paddle
import paddle.fluid as fluid
paddle.enable_static()
x = fluid.data(name="x", shape=[4, 4, 3], dtype="float32")
# x shape is [4, 4, 3]
out = fluid.layers.flatten(x=x, axis=2)
# out shape is [16, 3]