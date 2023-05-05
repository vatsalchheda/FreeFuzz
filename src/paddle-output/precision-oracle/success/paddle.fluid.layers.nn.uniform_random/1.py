results = dict()
import paddle
import time
arg_1_0 = 2
arg_1_1 = 1
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "float32"
arg_3 = -51.0
arg_4 = -78
arg_5 = 53
start = time.time()
results["time_low"] = paddle.fluid.layers.nn.uniform_random(arg_1,dtype=arg_2,min=arg_3,max=arg_4,seed=arg_5,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.fluid.layers.nn.uniform_random(arg_1,dtype=arg_2,min=arg_3,max=arg_4,seed=arg_5,)
results["time_high"] = time.time() - start

print(results)
