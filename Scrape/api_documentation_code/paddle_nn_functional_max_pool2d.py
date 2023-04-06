import paddle
import paddle.nn.functional as F

# max pool2d
x = paddle.uniform([1, 3, 32, 32], paddle.float32)
out = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
# output.shape [1, 3, 16, 16]
# for return_mask=True
out, max_indices = F.max_pool2d(x, kernel_size=2, stride=2, padding=0, return_mask=True)
# out.shape [1, 3, 16, 16], max_indices.shape [1, 3, 16, 16],