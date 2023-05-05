results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[2, 6], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = -52.0
arg_2_1 = -38.0
arg_2_2 = -12.0
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_low"] = paddle.searchsorted(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.int32)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_high"] = paddle.searchsorted(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
