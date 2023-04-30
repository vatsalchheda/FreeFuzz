results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16384,2048,[2, 3, 8, 8, 8], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024,2048,[0, 3, 3, 0, 52], dtype=paddle.bfloat16)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.nn.functional.conv3d(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.bfloat16)
start = time.time()
results["time_high"] = paddle.nn.functional.conv3d(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
