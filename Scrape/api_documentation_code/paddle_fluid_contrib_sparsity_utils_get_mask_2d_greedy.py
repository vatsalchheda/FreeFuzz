import numpy as np
import paddle.fluid.contrib.sparsity as sparsity

mat = np.array([[9, 8, 3, 7],
                [9, 2, 1, 10],
                [5, 1, 3, 6],
                [2, 4, 6, 1]])
mask = sparsity.get_mask_2d_greedy(mat, 2, 4)
# nparray([[1. 1. 0. 0.]
#          [1. 0. 0. 1.]
#          [0. 0. 1. 1.]
#          [0. 1. 1. 0.]])
sparsity.check_mask_2d(mask, 2, 4) # True