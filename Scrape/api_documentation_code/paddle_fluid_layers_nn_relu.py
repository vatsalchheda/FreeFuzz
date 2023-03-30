import paddle.fluid as fluid
import numpy as np
in1 = np.array([[-1,0],[1,2.6]])
with fluid.dygraph.guard():
    x1 = fluid.dygraph.to_variable(in1)
    out1 = fluid.layers.relu(x1)
    print(out1.numpy())
    # [[0.  0. ]
    #  [1.  2.6]]