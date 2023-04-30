results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-32768,2048,[3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = "Categorical_log_prob"
start = time.time()
results["time_low"] = paddle.log(arg_1,name=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.log(arg_1,name=arg_2,)
results["time_high"] = time.time() - start

print(results)
