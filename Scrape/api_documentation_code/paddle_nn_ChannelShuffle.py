import paddle
import paddle.nn as nn
x = paddle.arange(0, 0.6, 0.1, 'float32')
x = paddle.reshape(x, [1, 6, 1, 1])
# [[[[0.        ]],
#   [[0.10000000]],
#   [[0.20000000]],
#   [[0.30000001]],
#   [[0.40000001]],
#   [[0.50000000]]]]
channel_shuffle = nn.ChannelShuffle(3)
y = channel_shuffle(x)
# [[[[0.        ]],
#   [[0.20000000]],
#   [[0.40000001]],
#   [[0.10000000]],
#   [[0.30000001]],
#   [[0.50000000]]]]