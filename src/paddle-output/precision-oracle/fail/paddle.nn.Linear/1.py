results = dict()
import paddle
import time
arg_1 = "max"
arg_2 = 1024
arg_class = paddle.nn.Linear(arg_1,arg_2,)
arg_3_0_tensor = paddle.randint(-32,1,[1, 13], dtype=paddle.float16)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().type(paddle.float32)
arg_3 = [arg_3_0,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
