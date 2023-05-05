results = dict()
import paddle
import time
arg_1 = 36
arg_2 = 0
start = time.time()
results["time_low"] = paddle.static.ipu_shard_guard(index=arg_1,stage=arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.static.ipu_shard_guard(index=arg_1,stage=arg_2,)
results["time_high"] = time.time() - start

print(results)
