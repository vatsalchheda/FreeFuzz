results = dict()
import paddle
import time
arg_class = paddle.nn.Mish()
arg_1_0_tensor = paddle.randint(-64,64,[3], dtype=paddle.int8)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
start = time.time()
results["time_low"] = arg_class(*arg_1)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.int16)
arg_1 = [arg_1_0,]
start = time.time()
results["time_high"] = arg_class(*arg_1)
results["time_high"] = time.time() - start

print(results)
