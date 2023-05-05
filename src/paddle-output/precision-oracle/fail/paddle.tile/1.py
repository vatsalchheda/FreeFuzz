results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[3], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2_1 = 2
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = paddle.tile(arg_1,repeat_times=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.tile(arg_1,repeat_times=arg_2,)
results["time_high"] = time.time() - start

print(results)
