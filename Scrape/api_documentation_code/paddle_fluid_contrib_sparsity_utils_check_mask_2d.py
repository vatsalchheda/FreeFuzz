import numpy as np
import paddle.fluid.contrib.sparsity as sparsity

x = np.array([[0, 8, 9, 0],
              [9, 0, 0, 10],
              [5, 0, 0, 6],
              [0, 4, 6, 0]])
sparsity.check_mask_2d(x, 2, 4) # True

x = np.array([[0, 8, 0, 9],
              [9, 0, 0, 10],
              [0, 5, 0, 6],
              [0, 4, 6, 0]])
sparsity.check_mask_2d(x, 2, 4) # False

# x would be padded to shape (8, 8)
x = np.array([[0, 8, 0, 9],
              [9, 0, 7, 0],
              [0, 5, 0, 6],
              [3, 0, 6, 0],
              [1, 1, 0, 1]])
sparsity.check_mask_2d(x, 2, 4) # True