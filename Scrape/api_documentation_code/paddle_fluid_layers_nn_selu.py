import paddle
import paddle.fluid as fluid
import numpy as np
paddle.enable_static()

inputs = fluid.layers.data(name="x", shape=[2, 2], dtype="float32")
output = fluid.layers.selu(inputs)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

img = np.array([[0, 1],[2, 3]]).astype(np.float32)

res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
print(res) # [array([[0.      , 1.050701],[2.101402, 3.152103]], dtype=float32)]