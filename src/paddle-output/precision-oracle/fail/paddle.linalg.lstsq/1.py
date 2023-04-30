results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-2048,2048,[3, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,1024,[3, 3], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3 = "gelsd"
start = time.time()
results["time_low"] = paddle.linalg.lstsq(arg_1,arg_2,driver=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.linalg.lstsq(arg_1,arg_2,driver=arg_3,)
results["time_high"] = time.time() - start

print(results)
