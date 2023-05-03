import paddle
paddle.enable_static()
import paddle.fluid as fluid
x = fluid.layers.data(name='x', shape=[6, 10], lod_level=1)
out = fluid.layers.lod_append(x, [1,1,1,1,1,1])