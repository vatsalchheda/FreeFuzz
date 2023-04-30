results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8192,16,[3, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = False
start = time.time()
results["time_low"] = paddle.linalg.lu(arg_1,get_infos=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
start = time.time()
results["time_high"] = paddle.linalg.lu(arg_1,get_infos=arg_2,)
results["time_high"] = time.time() - start

print(results)
