import paddle.fluid as fluid
import paddle
import numpy as np
paddle.enable_static()

input_brelu = np.array([[-1,6],[1,15.6]])
with fluid.dygraph.guard():
    x = fluid.dygraph.to_variable(input_brelu)
    y = fluid.layers.brelu(x, t_min=1.0, t_max=10.0)
    print(y.numpy())
    #[[ 1.  6.]
    #[ 1. 10.]]