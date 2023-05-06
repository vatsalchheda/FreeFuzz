results = dict()
import paddle
import time
arg_1 = 1
arg_2 = 14
arg_3 = 113
arg_4 = 3
start = time.time()
results["time_low"] = paddle.profiler.make_scheduler(closed=arg_1,ready=arg_2,record=arg_3,repeat=arg_4,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.profiler.make_scheduler(closed=arg_1,ready=arg_2,record=arg_3,repeat=arg_4,)
results["time_high"] = time.time() - start

print(results)
