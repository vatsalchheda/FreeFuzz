results = dict()
import paddle
import time
arg_1_0 = "gaussian"
arg_1_1 = 7
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 16
start = time.time()
results["time_low"] = paddle.audio.functional.get_window(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.audio.functional.get_window(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
