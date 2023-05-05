results = dict()
import paddle
import time
arg_1 = 1.0
arg_2 = 0.5
arg_3 = 56
arg_4 = True
arg_5 = 3
arg_class = paddle.fluid.dygraph.learning_rate_scheduler.ReduceLROnPlateau(learning_rate=arg_1,decay_rate=arg_2,patience=arg_3,verbose=arg_4,cooldown=arg_5,)
arg_6 = []
start = time.time()
results["time_low"] = arg_class(*arg_6)
results["time_low"] = time.time() - start
arg_6 = []
start = time.time()
results["time_high"] = arg_class(*arg_6)
results["time_high"] = time.time() - start

print(results)
