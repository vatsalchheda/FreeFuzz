import paddle
paddle.enable_static()
import paddle.fluid as fluid
data = fluid.data(
    name="data", shape=[None, 3, 112, 112], dtype="float32")
lrn = fluid.layers.lrn(input=data)
print(lrn.shape)  # [-1, 3, 112, 112]
print(lrn.dtype)  # float32