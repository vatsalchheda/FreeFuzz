import numpy as np
import paddle.fluid.contrib.sparsity as sparsity

mat = np.array([[0, 1, 5, 4],
                [2, 7, 3, 6]])
mask = sparsity.get_mask_1d(mat, 2, 4)
# nparray([[0, 0, 1, 1],
#          [0, 1, 0, 1]])
sparsity.check_mask_1d(mask, 2, 4) # True