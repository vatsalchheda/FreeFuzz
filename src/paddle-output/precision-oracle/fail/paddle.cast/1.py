results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-128,128,[0, 3], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
arg_2 = "int64"
start = time.time()
results["time_low"] = paddle.cast(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.complex128)
start = time.time()
results["time_high"] = paddle.cast(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
