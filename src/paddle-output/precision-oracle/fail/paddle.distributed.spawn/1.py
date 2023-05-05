results = dict()
import paddle
import time
arg_1 = "train"
start = time.time()
results["time_low"] = paddle.distributed.spawn(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.distributed.spawn(arg_1,)
results["time_high"] = time.time() - start

print(results)
