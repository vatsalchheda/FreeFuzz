results = dict()
import paddle
import time
arg_1 = "leaky_relu"
arg_2 = 1.0
start = time.time()
results["time_low"] = paddle.nn.initializer.calculate_gain(arg_1,param=arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.initializer.calculate_gain(arg_1,param=arg_2,)
results["time_high"] = time.time() - start

print(results)
