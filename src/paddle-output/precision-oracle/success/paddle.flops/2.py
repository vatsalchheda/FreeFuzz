results = dict()
import paddle
import time
arg_1 = "__main__LeNetMultiInput"
arg_2_0 = -31
arg_2_1 = -33
arg_2_2 = 6
arg_2_3 = 31
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3 = "builtinsdict"
arg_4 = True
start = time.time()
results["time_low"] = paddle.flops(arg_1,arg_2,custom_ops=arg_3,print_detail=arg_4,)
results["time_low"] = time.time() - start
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
start = time.time()
results["time_high"] = paddle.flops(arg_1,arg_2,custom_ops=arg_3,print_detail=arg_4,)
results["time_high"] = time.time() - start

print(results)
