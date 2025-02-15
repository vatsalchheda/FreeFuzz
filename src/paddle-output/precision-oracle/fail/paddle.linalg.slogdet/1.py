results = dict()
import paddle
import time
real = paddle.rand([0, 3, 3], paddle.float32)
imag = paddle.rand([0, 3, 3], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
start = time.time()
results["time_low"] = paddle.linalg.slogdet(arg_1,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.complex64)
start = time.time()
results["time_high"] = paddle.linalg.slogdet(arg_1,)
results["time_high"] = time.time() - start

print(results)
