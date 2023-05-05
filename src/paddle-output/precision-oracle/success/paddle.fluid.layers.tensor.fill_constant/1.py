results = dict()
import paddle
import time
arg_1 = 0.0
arg_2_0 = 256
arg_2_1 = 256
arg_2_2 = 11
arg_2_3 = 1
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3 = "float32"
arg_4 = False
arg_5 = None
start = time.time()
results["time_low"] = paddle.fluid.layers.tensor.fill_constant(value=arg_1,shape=arg_2,dtype=arg_3,force_cpu=arg_4,name=arg_5,)
results["time_low"] = time.time() - start
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
start = time.time()
results["time_high"] = paddle.fluid.layers.tensor.fill_constant(value=arg_1,shape=arg_2,dtype=arg_3,force_cpu=arg_4,name=arg_5,)
results["time_high"] = time.time() - start

print(results)
