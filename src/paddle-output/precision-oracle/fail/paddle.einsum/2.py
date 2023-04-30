results = dict()
import paddle
import time
arg_1 = "...jk->...kj"
arg_2_tensor = paddle.randint(-32768,4,[2, 3, 2], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.einsum(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.einsum(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
