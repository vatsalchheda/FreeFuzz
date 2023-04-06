import paddle.fluid as fluid
import numpy as np

in1 = np.array([[1, 2, 3],
                [4, 5, 6]])
in2 = np.array([[11, 12, 13],
                [14, 15, 16]])
in3 = np.array([[21, 22],
                [23, 24]])
with fluid.dygraph.guard():
    x1 = fluid.dygraph.to_variable(in1)
    x2 = fluid.dygraph.to_variable(in2)
    x3 = fluid.dygraph.to_variable(in3)
    # When the axis is negative, the real axis is (axis + Rank(x)).
    # As follows, axis is -1, Rank(x) is 2, the real axis is 1
    out1 = fluid.layers.concat(input=[x1, x2, x3], axis=-1)
    out2 = fluid.layers.concat(input=[x1, x2], axis=0)
    print(out1.numpy())
    # [[ 1  2  3 11 12 13 21 22]
    #  [ 4  5  6 14 15 16 23 24]]
    print(out2.numpy())
    # [[ 1  2  3]
    #  [ 4  5  6]
    #  [11 12 13]
    #  [14 15 16]]