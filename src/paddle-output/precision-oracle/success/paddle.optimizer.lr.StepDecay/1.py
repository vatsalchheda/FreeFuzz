results = dict()
import paddle
import time
arg_1 = -1024.0
arg_2 = 2
arg_3 = 0.1
arg_class = paddle.optimizer.lr.StepDecay(learning_rate=arg_1,step_size=arg_2,gamma=arg_3,)
arg_4 = []
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_4 = []
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)
