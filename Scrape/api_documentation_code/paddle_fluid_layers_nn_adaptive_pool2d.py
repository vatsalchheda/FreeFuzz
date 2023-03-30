# average adaptive pool2d
# suppose input data in shape of [N, C, H, W], `pool_size` is [m, n],
# output shape is [N, C, m, n], adaptive pool divide H and W dimensions
# of input data into m * n grids averagely and performs poolings in each
# grid to get output.
# adaptive average pool performs calculations as follow:
#
#     for i in range(m):
#         for j in range(n):
#             hstart = floor(i * H / m)
#             hend = ceil((i + 1) * H / m)
#             wstart = floor(i * W / n)
#             wend = ceil((i + 1) * W / n)
#             output[:, :, i, j] = avg(input[:, :, hstart: hend, wstart: wend])
#
import paddle
paddle.enable_static()
data = paddle.rand(shape=[1,3,32,32])
pool_out = paddle.fluid.layers.adaptive_pool2d(
                  input=data,
                  pool_size=[3, 3],
                  pool_type='avg')

# max adaptive pool2d
# suppose input data in shape of [N, C, H, W], `pool_size` is [m, n],
# output shape is [N, C, m, n], adaptive pool divide H and W dimensions
# of input data into m * n grids averagely and performs poolings in each
# grid to get output.
# adaptive average pool performs calculations as follow:
#
#     for i in range(m):
#         for j in range(n):
#             hstart = floor(i * H / m)
#             hend = ceil((i + 1) * H / m)
#             wstart = floor(i * W / n)
#             wend = ceil((i + 1) * W / n)
#             output[:, :, i, j] = max(input[:, :, hstart: hend, wstart: wend])
#
import paddle
data = paddle.rand(shape=[1,3,32,32])
pool_out = paddle.fluid.layers.adaptive_pool2d(
                  input=data,
                  pool_size=[3, 3],
                  pool_type='max')