import numpy as np
import paddle
import paddle.nn.functional as F

paddle.enable_static()
data = paddle.static.data(name='X', shape=[None, 1, 28, 28], dtype='float32')
conv2d = paddle.static.nn.conv2d(input=data, num_filters=6, filter_size=3)

with paddle.fluid.contrib.mixed_precision.bf16.amp_utils.bf16_guard():
    bn = paddle.static.nn.batch_norm(input=conv2d, act="relu")
    pool = F.max_pool2d(bn, kernel_size=2, stride=2)
    hidden = paddle.static.nn.fc(pool, size=10)
    loss = paddle.mean(hidden)