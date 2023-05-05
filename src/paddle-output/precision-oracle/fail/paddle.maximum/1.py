results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([257], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(0,2,[1, 1])
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.maximum(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
arg_2 = arg_2_tensor.clone().astype(paddle.bool)
start = time.time()
results["time_high"] = paddle.maximum(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
