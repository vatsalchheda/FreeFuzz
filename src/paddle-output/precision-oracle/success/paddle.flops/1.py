results = dict()
import paddle
import time
arg_1 = "__main__LeNet"
arg_2 = 16
arg_3 = "builtinsdict"
arg_4 = False
start = time.time()
results["time_low"] = paddle.flops(arg_1,arg_2,custom_ops=arg_3,print_detail=arg_4,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.flops(arg_1,arg_2,custom_ops=arg_3,print_detail=arg_4,)
results["time_high"] = time.time() - start

print(results)
