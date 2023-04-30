results = dict()
import paddle
import time
arg_1 = 1024.0
arg_2_0 = 1
arg_2_1 = 3
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "paddleVarType"
arg_4 = False
arg_5 = None
start = time.time()
results["time_low"] = paddle.fluid.layers.tensor.fill_constant(value=arg_1,shape=arg_2,dtype=arg_3,force_cpu=arg_4,name=arg_5,)
results["time_low"] = time.time() - start
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.fluid.layers.tensor.fill_constant(value=arg_1,shape=arg_2,dtype=arg_3,force_cpu=arg_4,name=arg_5,)
results["time_high"] = time.time() - start

print(results)
