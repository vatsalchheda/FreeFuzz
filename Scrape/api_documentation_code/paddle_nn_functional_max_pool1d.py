import paddle
import paddle.nn.functional as F

data = paddle.uniform([1, 3, 32], paddle.float32)
pool_out = F.max_pool1d(data, kernel_size=2, stride=2, padding=0)
# pool_out shape: [1, 3, 16]
pool_out, indices = F.max_pool1d(data, kernel_size=2, stride=2, padding=0, return_mask=True)
# pool_out shape: [1, 3, 16],  indices shape: [1, 3, 16]