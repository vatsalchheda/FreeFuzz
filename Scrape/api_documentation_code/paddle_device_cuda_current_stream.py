# required: gpu
import paddle

s1 = paddle.device.cuda.current_stream()

s2 = paddle.device.cuda.current_stream(0)

s3 = paddle.device.cuda.current_stream(paddle.CUDAPlace(0))