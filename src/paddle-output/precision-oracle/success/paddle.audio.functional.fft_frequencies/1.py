results = dict()
import paddle
import time
arg_1 = 16000
arg_2 = 128
start = time.time()
results["time_low"] = paddle.audio.functional.fft_frequencies(arg_1,arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.audio.functional.fft_frequencies(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
