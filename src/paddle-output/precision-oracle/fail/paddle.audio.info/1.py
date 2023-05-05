results = dict()
import paddle
import time
arg_1 = "C:\Users\phalt\Desktop\FreeFuzz\test.wav"
start = time.time()
results["time_low"] = paddle.audio.info(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.audio.info(arg_1,)
results["time_high"] = time.time() - start

print(results)
