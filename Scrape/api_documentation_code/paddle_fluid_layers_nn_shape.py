import paddle.fluid as fluid
import numpy as np
import paddle
paddle.enable_static()

inputs = fluid.data(name="x", shape=[3, 100, 100], dtype="float32")
output = fluid.layers.shape(inputs)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

img = np.ones((3, 100, 100)).astype(np.float32)

res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
print(res) # [array([  3, 100, 100], dtype=int32)]