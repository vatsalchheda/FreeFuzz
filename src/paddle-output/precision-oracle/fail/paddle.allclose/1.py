results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8192,2,[2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768,4096,[2], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3 = 1e-05
arg_4 = 21.00000001
arg_5 = True
arg_6 = "ignore_nan"
start = time.time()
results["time_low"] = paddle.allclose(arg_1,arg_2,rtol=arg_3,atol=arg_4,equal_nan=arg_5,name=arg_6,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.allclose(arg_1,arg_2,rtol=arg_3,atol=arg_4,equal_nan=arg_5,name=arg_6,)
results["time_high"] = time.time() - start

print(results)
