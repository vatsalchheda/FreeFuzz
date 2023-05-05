results = dict()
import paddle
import time
arg_1_0 = 5
arg_1_1 = 8
arg_1 = [arg_1_0,arg_1_1,]
arg_2_0 = False
arg_2_1 = "max"
arg_2_2 = True
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_class = paddle.optimizer.lr.PiecewiseDecay(boundaries=arg_1,values=arg_2,)
arg_3 = []
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3 = []
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
