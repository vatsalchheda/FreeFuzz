import paddle.fluid as fluid
import numpy as np
import paddle
paddle.enable_static()

place = fluid.core.CPUPlace()

x = fluid.data(name="x", shape=[2,2], dtype="int32", lod_level=1)
res = fluid.layers.hash(name="res", input=x, hash_size=1000, num_hash=4)

exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
in1 = np.array([[1,2],[3,4]]).astype("int32")
print(in1)
x_i = fluid.create_lod_tensor(in1, [[0, 2]], place)
res = exe.run(fluid.default_main_program(), feed={'x':x_i}, fetch_list=[res], return_numpy=False)
print(np.array(res[0]))
# [[[722]
#   [407]
#   [337]
#   [395]]
#  [[603]
#   [590]
#   [386]
#   [901]]]