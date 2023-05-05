results = dict()
import paddle
import time
arg_1 = 0.5
arg_2 = -4.0
arg_class = paddle.fluid.dygraph.learning_rate_scheduler.LambdaDecay(arg_1,lr_lambda=arg_2,)
arg_3 = []
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3 = []
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
