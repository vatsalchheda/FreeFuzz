results = dict()
import paddle
import time
arg_1 = 64
arg_2 = -12.5
arg_3 = 10000
arg_4 = True
arg_5 = "float64"
start = time.time()
results["time_low"] = paddle.audio.functional.mel_frequencies(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.audio.functional.mel_frequencies(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_high"] = time.time() - start

print(results)
