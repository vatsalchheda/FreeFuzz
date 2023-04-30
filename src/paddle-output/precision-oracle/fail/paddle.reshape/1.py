results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1024,1024,[2, 4, 6], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_0_tensor = paddle.randint(-16,4,[1], dtype=paddle.int8)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
start = time.time()
results["time_low"] = paddle.reshape(arg_1,shape=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2_0 = arg_2_0_tensor.clone().type(paddle.int32)
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = paddle.reshape(arg_1,shape=arg_2,)
results["time_high"] = time.time() - start

print(results)
