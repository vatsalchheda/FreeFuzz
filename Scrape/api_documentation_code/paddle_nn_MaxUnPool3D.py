import paddle
import paddle.nn.functional as F

data = paddle.rand(shape=[1, 1, 4, 4, 6])
pool_out, indices = F.max_pool3d(data, kernel_size=2, stride=2, padding=0, return_mask=True)
# pool_out shape: [1, 1, 2, 2, 3],  indices shape: [1, 1, 2, 2, 3]
Unpool3D = paddle.nn.MaxUnPool3D(kernel_size=2, padding=0)
unpool_out = Unpool3D(pool_out, indices)
# unpool_out shape: [1, 1, 4, 4, 6]