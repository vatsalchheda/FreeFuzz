results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1024,8,[2, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768,1024,[2, 2], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3 = False
start = time.time()
results["time_low"] = paddle.tensordot(arg_1,arg_2,axes=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
arg_2 = arg_2_tensor.clone().type(paddle.float64)
start = time.time()
results["time_high"] = paddle.tensordot(arg_1,arg_2,axes=arg_3,)
results["time_high"] = time.time() - start

print(results)
