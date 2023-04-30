results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1,128,[1], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8,4096,[1], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.Tensor.subtract_(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.Tensor.subtract_(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
