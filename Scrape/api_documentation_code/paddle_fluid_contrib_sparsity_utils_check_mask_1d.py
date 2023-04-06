import numpy as np
import paddle.fluid.contrib.sparsity as sparsity

x = np.array([[0, 1, 3, 0],
              [1, 0, 0, 1]])
sparsity.check_mask_1d(x, 2, 4) # True

x = np.array([[0, 1, 5, 4],
              [1, 0, 0, 1]])
sparsity.check_mask_1d(x, 2, 4) # False

# x would be padded to shape (2, 8)
x = np.array([[0, 1, 0, 4, 6],
              [1, 0, 0, 1, 7]])
sparsity.check_mask_1d(x, 2, 4) # True