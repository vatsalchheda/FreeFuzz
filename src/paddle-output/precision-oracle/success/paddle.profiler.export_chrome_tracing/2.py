results = dict()
import paddle
import time
arg_1 = "./log"
start = time.time()
results["time_low"] = paddle.profiler.export_chrome_tracing(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.profiler.export_chrome_tracing(arg_1,)
results["time_high"] = time.time() - start

print(results)
