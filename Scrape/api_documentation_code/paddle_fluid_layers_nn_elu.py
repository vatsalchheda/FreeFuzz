import paddle.fluid as fluid
import numpy as np

input_elu = np.array([[-1,6],[1,15.6]])
with fluid.dygraph.guard():
    x = fluid.dygraph.to_variable(input_elu)
    y = fluid.layers.elu(x, alpha=0.2)
    print(y.numpy())
    # [[-0.12642411  6.        ]
    # [ 1.          15.6       ]]