results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-512,8,[1], dtype=paddle.bfloat16)
arg_1 = arg_1_tensor.clone()
start = time.time()
results["time_low"] = paddle.Tensor.item(arg_1,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.bfloat16)
start = time.time()
results["time_high"] = paddle.Tensor.item(arg_1,)
results["time_high"] = time.time() - start

print(results)
