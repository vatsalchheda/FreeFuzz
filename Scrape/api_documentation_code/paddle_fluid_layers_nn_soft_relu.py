import paddle.fluid as fluid
import numpy as np
import numpy as np
import paddle

paddle.enable_static()
inputs = fluid.layers.data(name="x", shape=[2, 2], dtype="float32")
output = fluid.layers.soft_relu(inputs, threshold=20.0)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

img = np.array([[0, 1],[2, 3]]).astype(np.float32)

res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
print(res) # [array([[0.6931472, 1.3132616], [2.126928 , 3.0485873]], dtype=float32)]