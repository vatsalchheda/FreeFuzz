import numpy as np
import paddle.fluid.contrib.sparsity as sparsity

mat = np.array([[2, 8, 9, 9],
                [9, 1, 3, 9],
                [5, 6, 3, 9],
                [2, 4, 6, 9]])
mask_greedy = sparsity.get_mask_2d_greedy(mat, 2, 4)
mask_best = sparsity.get_mask_2d_best(mat, 2, 4)
print("L1 norm of `greedy` sparse matrix", np.multiply(mat, mask_greedy).sum()) # 56
print("L1 norm of `best` sparse matrix", np.multiply(mat, mask_best).sum()) # 61