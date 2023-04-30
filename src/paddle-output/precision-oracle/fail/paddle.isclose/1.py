results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8,64,[2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2048,32768,[2], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3 = 18.00001
arg_4 = 56.00000001
arg_5 = "mean"
arg_6 = "ignore_nan"
start = time.time()
results["time_low"] = paddle.isclose(arg_1,arg_2,rtol=arg_3,atol=arg_4,equal_nan=arg_5,name=arg_6,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.isclose(arg_1,arg_2,rtol=arg_3,atol=arg_4,equal_nan=arg_5,name=arg_6,)
results["time_high"] = time.time() - start

print(results)
