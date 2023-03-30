import paddle.fluid as fluid
import numpy as np
import paddle
paddle.enable_static()
data = fluid.data(name="x", shape=[-1, 3], dtype="float32")
label = fluid.data(name="y", shape=[-1, 3], dtype="float32")
result = fluid.layers.smooth_l1(data,label)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
x = np.random.rand(3,3).astype("float32")
y = np.random.rand(3,3).astype("float32")
output= exe.run(feed={"x":x, "y":y},
                 fetch_list=[result])
print(output)

#[array([[0.08220536],
#       [0.36652038],
#      [0.20541131]], dtype=float32)]