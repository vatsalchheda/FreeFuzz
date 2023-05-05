results = dict()
import paddle
import time
arg_1 = "mean"
arg_2 = 1
arg_3 = 4
arg_4 = 31
start = time.time()
results["time_low"] = paddle.profiler.make_scheduler(closed=arg_1,ready=arg_2,record=arg_3,skip_first=arg_4,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.profiler.make_scheduler(closed=arg_1,ready=arg_2,record=arg_3,skip_first=arg_4,)
results["time_high"] = time.time() - start

print(results)
