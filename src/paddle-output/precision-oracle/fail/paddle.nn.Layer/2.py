results = dict()
import paddle
import time
arg_class = paddle.nn.Layer()
arg_1_0_tensor = paddle.randint(-16,128,[1, 1, 2, 2, 3], dtype=paddle.int8)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-32,32,[1, 1, 2, 0, 3], dtype=paddle.int8)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_low"] = arg_class(*arg_1)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.int32)
arg_1_1 = arg_1_1_tensor.clone().type(paddle.int32)
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = arg_class(*arg_1)
results["time_high"] = time.time() - start

print(results)
