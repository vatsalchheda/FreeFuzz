results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16,128,[3, 4], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128,4,[3, 2], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-2048,4096,[53, 9], dtype=paddle.float16)
arg_3 = arg_3_tensor.clone()
start = time.time()
results["time_low"] = paddle.scatter_nd_add(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = arg_2_tensor.clone().type(paddle.int64)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.scatter_nd_add(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
