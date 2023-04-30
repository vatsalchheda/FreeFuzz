results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-32768,1024,[3, 1, 7, 112, 112], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.2
start = time.time()
results["time_low"] = paddle.nn.functional.elu(arg_1,alpha=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.elu(arg_1,alpha=arg_2,)
results["time_high"] = time.time() - start

print(results)
