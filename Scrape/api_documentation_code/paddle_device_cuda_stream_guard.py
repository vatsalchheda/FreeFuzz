# required: gpu
import paddle

s = paddle.device.cuda.Stream()
data1 = paddle.ones(shape=[20])
data2 = paddle.ones(shape=[20])
with paddle.device.cuda.stream_guard(s):
    data3 = data1 + data2