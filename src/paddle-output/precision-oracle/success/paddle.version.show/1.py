results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.version.show()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.version.show()
results["time_high"] = time.time() - start

print(results)
