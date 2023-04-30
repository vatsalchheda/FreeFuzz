results = dict()
import paddle
import time
arg_1 = True
arg_2 = "Beta"
start = time.time()
results["time_low"] = paddle.distribution.register_kl(arg_1,arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.distribution.register_kl(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
