import paddle
paddle.enable_static()
import paddle.fluid as fluid
import numpy as np
place = fluid.CPUPlace()
theta = fluid.data(name="x", shape=[None, 2, 3], dtype="float32")
out_shape = fluid.data(name="y", shape=[4], dtype="int32")
grid_0 = fluid.layers.affine_grid(theta, out_shape)
grid_1 = fluid.layers.affine_grid(theta, [5, 3, 28, 28])
batch_size=2
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
output= exe.run(feed={"x": np.random.rand(batch_size,2,3).astype("float32"),
                      "y": np.array([5, 3, 28, 28]).astype("int32")},
                      fetch_list=[grid_0.name, grid_1.name])
print(output[0])
print(output[1])