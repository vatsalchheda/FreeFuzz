results = dict()
import paddle
import time
real = paddle.rand([36, 4], paddle.float32)
imag = paddle.rand([36, 4], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
start = time.time()
results["time_low"] = paddle.nn.functional.channel_shuffle(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.complex64)
start = time.time()
results["time_high"] = paddle.nn.functional.channel_shuffle(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
