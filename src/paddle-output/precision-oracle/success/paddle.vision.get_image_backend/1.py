results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.vision.get_image_backend()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.vision.get_image_backend()
results["time_high"] = time.time() - start

print(results)
