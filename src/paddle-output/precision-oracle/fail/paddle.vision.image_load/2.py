results = dict()
import paddle
import time
arg_1 = "temp.png"
start = time.time()
results["time_low"] = paddle.vision.image_load(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.vision.image_load(arg_1,)
results["time_high"] = time.time() - start

print(results)
