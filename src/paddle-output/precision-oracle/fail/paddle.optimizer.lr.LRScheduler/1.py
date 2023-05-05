results = dict()
import paddle
import time
arg_1 = 52.0002
arg_2 = -1
arg_3 = True
arg_class = paddle.optimizer.lr.LRScheduler(arg_1,arg_2,arg_3,)
arg_4 = []
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_4 = []
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)
