import paddle.fluid as fluid
import numpy as np

in1 = np.array([[[5,8,9,5],
                [0,0,1,7],
                [6,9,2,4]],
                [[5,2,4,2],
                [4,7,7,9],
                [1,7,0,6]]]).astype(np.float32)
with fluid.dygraph.guard():
    x = fluid.dygraph.to_variable(in1)
    out1 = fluid.layers.argsort(input=x, axis=-1)
    out2 = fluid.layers.argsort(input=x, axis=0)
    out3 = fluid.layers.argsort(input=x, axis=1)
    print(out1[0].numpy())
    # [[[5. 5. 8. 9.]
    #   [0. 0. 1. 7.]
    #   [2. 4. 6. 9.]]
    #  [[2. 2. 4. 5.]
    #   [4. 7. 7. 9.]
    #   [0. 1. 6. 7.]]]
    print(out1[1].numpy())
    # [[[0 3 1 2]
    #   [0 1 2 3]
    #   [2 3 0 1]]
    #  [[1 3 2 0]
    #   [0 1 2 3]
    #   [2 0 3 1]]]
    print(out2[0].numpy())
    # [[[5. 2. 4. 2.]
    #   [0. 0. 1. 7.]
    #   [1. 7. 0. 4.]]
    #  [[5. 8. 9. 5.]
    #   [4. 7. 7. 9.]
    #   [6. 9. 2. 6.]]]
    print(out3[0].numpy())
    # [[[0. 0. 1. 4.]
    #   [5. 8. 2. 5.]
    #   [6. 9. 9. 7.]]
    #  [[1. 2. 0. 2.]
    #   [4. 7. 4. 6.]
    #   [5. 7. 7. 9.]]]