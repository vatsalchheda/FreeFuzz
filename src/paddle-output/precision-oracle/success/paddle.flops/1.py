results = dict()
import paddle
import time
arg_1 = "__main__LeNet"
arg_2_0 = -30.0
arg_2_1 = False
arg_2_2 = False
arg_2_3 = -54.0
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3 = "mean"
arg_4 = 1e+20
start = time.time()
results["time_low"] = paddle.flops(arg_1,arg_2,custom_ops=arg_3,print_detail=arg_4,)
results["time_low"] = time.time() - start
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
start = time.time()
results["time_high"] = paddle.flops(arg_1,arg_2,custom_ops=arg_3,print_detail=arg_4,)
results["time_high"] = time.time() - start

print(results)
