results = dict()
import paddle
import time
arg_1 = "sum"
start = time.time()
results["time_low"] = paddle.audio.load(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.audio.load(arg_1,)
results["time_high"] = time.time() - start

print(results)
