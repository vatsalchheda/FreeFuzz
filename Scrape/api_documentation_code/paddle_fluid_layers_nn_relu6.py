import paddle.fluid as fluid
import numpy as np
in1 = np.array([[-1,0],[2.5,7.8]])
with fluid.dygraph.guard():
    x1 = fluid.dygraph.to_variable(in1)
    out1 = fluid.layers.relu6(x=x1, threshold=6.0)
    print(out1.numpy())
    # [[0.  0. ]
    #  [2.5 6. ]]