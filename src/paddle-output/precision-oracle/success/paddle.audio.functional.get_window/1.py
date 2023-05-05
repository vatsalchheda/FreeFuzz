results = dict()
import paddle
import time
arg_1 = "cosine"
arg_2 = 512
start = time.time()
results["time_low"] = paddle.audio.functional.get_window(arg_1,arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.audio.functional.get_window(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
