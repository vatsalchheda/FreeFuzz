results = dict()
import paddle
import time
arg_1 = 5
arg_class = paddle.nn.Upsample(size=arg_1,)
arg_2 = []
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_2 = []
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)
