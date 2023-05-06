results = dict()
import paddle
import time
arg_1 = True
arg_2 = 3
arg_3 = True
start = time.time()
results["time_low"] = paddle.amp.auto_cast(enable=arg_1,custom_white_list=arg_2,level=arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.amp.auto_cast(enable=arg_1,custom_white_list=arg_2,level=arg_3,)
results["time_high"] = time.time() - start

print(results)
