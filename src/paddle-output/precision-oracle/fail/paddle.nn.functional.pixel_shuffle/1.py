results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-32768,512,[2, 9, 4, 4], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 14
start = time.time()
results["time_low"] = paddle.nn.functional.pixel_shuffle(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.pixel_shuffle(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
