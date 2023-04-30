results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-4,4096,[3, 224, 224, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
start = time.time()
results["time_low"] = paddle.nn.functional.softmax(arg_1,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.softmax(arg_1,)
results["time_high"] = time.time() - start

print(results)
