results = dict()
import paddle
import time
arg_1 = True
start = time.time()
results["time_low"] = paddle.audio.backends.set_backend(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.audio.backends.set_backend(arg_1,)
results["time_high"] = time.time() - start

print(results)
