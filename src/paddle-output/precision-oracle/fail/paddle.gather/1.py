results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-128,64,[3, 2], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,16,[2], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3 = -5
start = time.time()
results["time_low"] = paddle.gather(arg_1,arg_2,axis=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = arg_2_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.gather(arg_1,arg_2,axis=arg_3,)
results["time_high"] = time.time() - start

print(results)
