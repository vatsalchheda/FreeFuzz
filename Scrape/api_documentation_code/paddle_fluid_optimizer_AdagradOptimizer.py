import numpy as np
import paddle.fluid as fluid

np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
inp = fluid.data(name="inp", shape=[2, 2])
out = fluid.layers.fc(inp, size=3)
out = fluid.layers.reduce_sum(out)
optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.2)
optimizer.minimize(out)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
exe.run(
    feed={"inp": np_inp},
    fetch_list=[out.name])