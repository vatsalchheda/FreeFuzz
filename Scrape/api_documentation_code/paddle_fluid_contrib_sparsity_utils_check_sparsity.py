import numpy as np
import paddle.fluid.contrib.sparsity as sparsity

tensor = np.array([[2, 8, 9, 9],
                   [9, 1, 3, 9],
                   [5, 6, 3, 9],
                   [2, 4, 6, 9]])
mask_1d = sparsity.create_mask(tensor, func_name=sparsity.MaskAlgo.MASK_1D)
# nparray([[0 0 1 1],
#          [1 0 0 1],
#          [0 1 0 1],
#          [0 0 1 1]])
sparsity.check_sparsity(mask_1d, func_name=sparsity.CheckMethod.CHECK_1D) # True
sparsity.check_sparsity(mask_1d, func_name=sparsity.CheckMethod.CHECK_2D) # False