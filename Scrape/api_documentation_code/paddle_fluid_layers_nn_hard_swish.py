import paddle.fluid as fluid
import paddle
import numpy as np
paddle.enable_static()

DATATYPE='float32'

x_data = np.array([i for i in range(1,5)]).reshape([1,1,4]).astype(DATATYPE)

x = fluid.data(name="x", shape=[None,1,4], dtype=DATATYPE)
y = fluid.layers.hard_swish(x)

place = fluid.CPUPlace()
#place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
out, = exe.run(feed={'x':x_data}, fetch_list=[y.name])
print(out)  # [[0.66666667, 1.66666667,3., 4.]]