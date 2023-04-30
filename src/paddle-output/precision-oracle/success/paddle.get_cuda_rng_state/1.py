results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.get_cuda_rng_state()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.get_cuda_rng_state()
results["time_high"] = time.time() - start

print(results)
