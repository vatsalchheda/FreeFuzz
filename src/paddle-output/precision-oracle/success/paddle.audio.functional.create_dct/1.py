results = dict()
import paddle
import time
arg_1 = 43
arg_2 = 51
start = time.time()
results["time_low"] = paddle.audio.functional.create_dct(arg_1,arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.audio.functional.create_dct(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
