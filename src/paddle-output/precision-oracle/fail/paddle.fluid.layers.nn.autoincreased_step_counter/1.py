results = dict()
import paddle
import time
arg_1 = "@LR_DECAY_COUNTER@"
arg_2 = 0
arg_3 = 1
start = time.time()
results["time_low"] = paddle.fluid.layers.nn.autoincreased_step_counter(counter_name=arg_1,begin=arg_2,step=arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.fluid.layers.nn.autoincreased_step_counter(counter_name=arg_1,begin=arg_2,step=arg_3,)
results["time_high"] = time.time() - start

print(results)
