import paddle
import paddle.nn.functional as F

data = paddle.rand(shape=[1, 3, 16])
pool_out, indices = F.max_pool1d(data, kernel_size=2, stride=2, padding=0, return_mask=True)
# pool_out shape: [1, 3, 8],  indices shape: [1, 3, 8]
unpool_out = F.max_unpool1d(pool_out, indices, kernel_size=2, padding=0)
# unpool_out shape: [1, 3, 16]