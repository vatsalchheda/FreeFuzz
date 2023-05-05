results = dict()
import paddle
import time
arg_1 = "roi_2"
arg_2_0 = None
arg_2_1 = -48
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "replicate"
arg_4 = -21
start = time.time()
results["time_low"] = paddle.fluid.data(name=arg_1,shape=arg_2,dtype=arg_3,lod_level=arg_4,)
results["time_low"] = time.time() - start
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.fluid.data(name=arg_1,shape=arg_2,dtype=arg_3,lod_level=arg_4,)
results["time_high"] = time.time() - start

print(results)
