import paddle
import paddle.fluid as fluid
import numpy as np
paddle.enable_static()

data = fluid.layers.data(name="data", shape=[-1,10], dtype='float64')
target_tensor = fluid.layers.data(
  name="target_tensor", shape=[-1,20], dtype='float64')
result = fluid.layers.expand_as(x=data, target_tensor=target_tensor)
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
x = np.random.rand(3,10)
y = np.random.rand(3,20)
output= exe.run(feed={"data":x,"target_tensor":y},fetch_list=[result.name])
print(output[0].shape)
#(3,20)