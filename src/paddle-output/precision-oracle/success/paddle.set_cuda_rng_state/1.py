results = dict()
import paddle
import time
arg_1 = []
start = time.time()
results["time_low"] = paddle.set_cuda_rng_state(arg_1,)
results["time_low"] = time.time() - start
arg_1 = []
start = time.time()
results["time_high"] = paddle.set_cuda_rng_state(arg_1,)
results["time_high"] = time.time() - start

print(results)
