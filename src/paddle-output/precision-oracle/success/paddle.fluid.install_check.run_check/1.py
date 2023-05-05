results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.fluid.install_check.run_check()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.fluid.install_check.run_check()
results["time_high"] = time.time() - start

print(results)
