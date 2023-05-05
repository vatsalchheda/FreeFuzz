results = dict()
import paddle
import time
real = paddle.rand([1, 30001], paddle.float32)
imag = paddle.rand([1, 30001], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = 5
start = time.time()
results["time_low"] = paddle.topk(arg_1,k=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.complex64)
start = time.time()
results["time_high"] = paddle.topk(arg_1,k=arg_2,)
results["time_high"] = time.time() - start

print(results)
