results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([10, 8, 22, 123], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -1024.0
start = time.time()
results["time_low"] = paddle.full_like(arg_1,fill_value=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.full_like(arg_1,fill_value=arg_2,)
results["time_high"] = time.time() - start

print(results)
