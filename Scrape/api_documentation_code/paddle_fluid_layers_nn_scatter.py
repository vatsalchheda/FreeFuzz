import paddle
import numpy as np
import paddle.fluid as fluid
paddle.enable_static()

input = fluid.layers.data(name='data', shape=[3, 2], dtype='float32', append_batch_size=False)
index = fluid.layers.data(name='index', shape=[4], dtype='int64', append_batch_size=False)
updates = fluid.layers.data(name='update', shape=[4, 2], dtype='float32', append_batch_size=False)

output = fluid.layers.scatter(input, index, updates, overwrite=False)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

in_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float32)
index_data = np.array([2, 1, 0, 1]).astype(np.int64)
update_data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype(np.float32)

res = exe.run(fluid.default_main_program(), feed={'data':in_data, "index":index_data, "update":update_data}, fetch_list=[output])
print(res)
# [array([[3., 3.],
#   [6., 6.],
#   [1., 1.]], dtype=float32)]