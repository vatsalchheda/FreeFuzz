results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8192,32,[53, 3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,8,[2], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3 = 0
arg_4_tensor = paddle.randint(-16384,64,[2, 3], dtype=paddle.float16)
arg_4 = arg_4_tensor.clone()
start = time.time()
results["time_low"] = paddle.index_add(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.int32)
arg_4 = arg_4_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.index_add(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)
