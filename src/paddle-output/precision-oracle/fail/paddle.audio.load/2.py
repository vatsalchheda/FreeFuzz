results = dict()
import paddle
import time
arg_1 = "C:\Users\phalt\.cache\paddle\dataset\ESC-50-master\audio\1-101296-A-19.wav"
start = time.time()
results["time_low"] = paddle.audio.load(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.audio.load(arg_1,)
results["time_high"] = time.time() - start

print(results)
