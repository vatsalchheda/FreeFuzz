results = dict()
import paddle
import time
arg_1 = 8
arg_2 = 125
arg_3 = 40.01
start = time.time()
results["time_low"] = paddle.fluid.layers.learning_rate_scheduler.noam_decay(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.fluid.layers.learning_rate_scheduler.noam_decay(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
