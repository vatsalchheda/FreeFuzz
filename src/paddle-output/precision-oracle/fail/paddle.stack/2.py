results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=128, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_0_tensor = int8_tensor
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
start = time.time()
results["time_low"] = paddle.stack(arg_1,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.int32)
arg_1 = [arg_1_0,]
start = time.time()
results["time_high"] = paddle.stack(arg_1,)
results["time_high"] = time.time() - start

print(results)
