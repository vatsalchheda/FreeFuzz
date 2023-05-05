results = dict()
import paddle
import time
real = paddle.rand([4, 4], paddle.float32)
imag = paddle.rand([4, 4], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
start = time.time()
results["time_low"] = paddle.angle(arg_1,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.complex64)
start = time.time()
results["time_high"] = paddle.angle(arg_1,)
results["time_high"] = time.time() - start

print(results)
