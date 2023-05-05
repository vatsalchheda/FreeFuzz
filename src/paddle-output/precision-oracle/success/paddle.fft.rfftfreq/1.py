results = dict()
import paddle
import time
arg_1 = 20
arg_2 = 0.5
arg_3 = "float32"
start = time.time()
results["time_low"] = paddle.fft.rfftfreq(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.fft.rfftfreq(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
