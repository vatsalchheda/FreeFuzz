import paddle
import paddle.nn as nn

input_data = paddle.randn(shape=(2,3,6,10)).astype(paddle.float32)
upsample_out = paddle.nn.Upsample(size=[12,12])

output = upsample_out(x=input_data)
print(output.shape)
# [2L, 3L, 12L, 12L]