results = dict()
import paddle
import time
arg_1 = "i,i->"
arg_2_tensor = paddle.randint(-128,4096,[4], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-1,16,[], dtype=paddle.int8)
arg_3 = arg_3_tensor.clone()
start = time.time()
results["time_low"] = paddle.einsum(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.int16)
start = time.time()
results["time_high"] = paddle.einsum(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
