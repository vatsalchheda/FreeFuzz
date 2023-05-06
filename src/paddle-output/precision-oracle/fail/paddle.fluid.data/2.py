results = dict()
import paddle
import time
arg_1 = False
arg_2_0 = -1
arg_2_1 = True
arg_2_2 = "zeros"
arg_2_3 = "replicate"
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3 = "float32"
start = time.time()
results["time_low"] = paddle.fluid.data(name=arg_1,shape=arg_2,dtype=arg_3,)
results["time_low"] = time.time() - start
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
start = time.time()
results["time_high"] = paddle.fluid.data(name=arg_1,shape=arg_2,dtype=arg_3,)
results["time_high"] = time.time() - start

print(results)
