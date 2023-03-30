# average adaptive pool3d
# suppose input data in shape of [N, C, D, H, W], `pool_size` is [l, m, n],
# output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimensions
# of input data into l * m * n grids averagely and performs poolings in each
# grid to get output.
# adaptive average pool performs calculations as follow:
#
#     for i in range(l):
#         for j in range(m):
#             for k in range(n):
#                 dstart = floor(i * D / l)
#                 dend = ceil((i + 1) * D / l)
#                 hstart = floor(j * H / m)
#                 hend = ceil((j + 1) * H / m)
#                 wstart = floor(k * W / n)
#                 wend = ceil((k + 1) * W / n)
#                 output[:, :, i, j, k] =
#                     avg(input[:, :, dstart:dend, hstart: hend, wstart: wend])
#

import paddle
paddle.enable_static()
data = paddle.rand(shape=[1,3,32,32,32])
pool_out = paddle.fluid.layers.adaptive_pool3d(
                  input=data,
                  pool_size=[3, 3, 3],
                  pool_type='avg')

# max adaptive pool3d
# suppose input data in shape of [N, C, D, H, W], `pool_size` is [l, m, n],
# output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimensions
# of input data into l * m * n grids averagely and performs poolings in each
# grid to get output.
# adaptive average pool performs calculations as follow:
#
#     for i in range(l):
#         for j in range(m):
#             for k in range(n):
#                 dstart = floor(i * D / l)
#                 dend = ceil((i + 1) * D / l)
#                 hstart = floor(j * H / m)
#                 hend = ceil((j + 1) * H / m)
#                 wstart = floor(k * W / n)
#                 wend = ceil((k + 1) * W / n)
#                 output[:, :, i, j, k] =
#                     avg(input[:, :, dstart:dend, hstart: hend, wstart: wend])
#

import paddle
data = paddle.rand(shape=[1,3,32,32,32])
pool_out = paddle.fluid.layers.adaptive_pool3d(
                  input=data,
                  pool_size=[3, 3, 3],
                  pool_type='max')