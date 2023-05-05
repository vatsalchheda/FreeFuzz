results = dict()
import paddle
import time
arg_1 = -63.0
arg_2 = 3
arg_class = paddle.fluid.dygraph.learning_rate_scheduler.StepDecay(arg_1,step_size=arg_2,)
arg_3 = []
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3 = []
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
