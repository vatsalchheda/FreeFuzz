results = dict()
import paddle
import time
arg_1 = 62.0
arg_2 = 1024
arg_3 = 0.01
start = time.time()
results["time_low"] = paddle.fluid.layers.learning_rate_scheduler.noam_decay(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.fluid.layers.learning_rate_scheduler.noam_decay(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
