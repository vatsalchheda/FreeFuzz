results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.distributed.init_parallel_env()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.distributed.init_parallel_env()
results["time_high"] = time.time() - start

print(results)
