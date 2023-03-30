import paddle.fluid as fluid
import numpy as np

in1 = np.array([[[5,8,9,5],
                [0,0,1,7],
                [6,9,2,4]],
                [[5,2,4,2],
                [4,7,7,9],
                [1,7,0,6]]])
with fluid.dygraph.guard():
    x = fluid.dygraph.to_variable(in1)
    out1 = fluid.layers.argmin(x=x, axis=-1)
    out2 = fluid.layers.argmin(x=x, axis=0)
    out3 = fluid.layers.argmin(x=x, axis=1)
    out4 = fluid.layers.argmin(x=x, axis=2)
    print(out1.numpy())
    # [[0 0 2]
    #  [1 0 2]]
    print(out2.numpy())
    # [[0 1 1 1]
    #  [0 0 0 0]
    #  [1 1 1 0]]
    print(out3.numpy())
    # [[1 1 1 2]
    #  [2 0 2 0]]
    print(out4.numpy())
    # [[0 0 2]
    #  [1 0 2]]