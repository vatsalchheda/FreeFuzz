# required: gpu
import paddle
import paddle.incubate as incubate

x = paddle.rand((1, 1, 32, 32))

rst = incubate.softmax_mask_fuse_upper_triangle(x)
# [[[[1.        , 0.        , 0.        , ..., 0., 0., 0.],
#    [0.45324376, 0.54675621, 0.        , ..., 0., 0., 0.],
#    [0.32674268, 0.28156221, 0.39169508, ..., 0., 0., 0.]
#     ... ]]]