results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=128, shape=[7], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1.0
arg_2_1 = 2.0
arg_2_2 = 3.0
arg_2_3 = 4.0
arg_2_4 = 5.0
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,arg_2_4,]
start = time.time()
results["time_low"] = paddle.searchsorted(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,arg_2_4,]
start = time.time()
results["time_high"] = paddle.searchsorted(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
