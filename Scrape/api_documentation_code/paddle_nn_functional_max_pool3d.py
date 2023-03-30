import paddle
import paddle.nn.functional as F

# max pool3d
x = paddle.uniform([1, 3, 32, 32, 32])
output = F.max_pool3d(x,
                      kernel_size=2,
                      stride=2, padding=0)
# output.shape [1, 3, 16, 16, 16]
# for return_mask=True
x = paddle.uniform([1, 3, 32, 32, 32])
output, max_indices = paddle.nn.functional.max_pool3d(x,
                                                      kernel_size=2,
                                                      stride=2,
                                                      padding=0,
                                                      return_mask=True)

# output.shape [1, 3, 16, 16, 16], max_indices.shape [1, 3, 16, 16, 16]