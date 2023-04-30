results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-4,4,[2, 4], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
start = time.time()
results["time_low"] = paddle.nanmean(arg_1,axis=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nanmean(arg_1,axis=arg_2,)
results["time_high"] = time.time() - start

print(results)
