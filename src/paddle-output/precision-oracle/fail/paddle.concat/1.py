results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[1, 49], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_0_tensor = int8_tensor
arg_1_0 = arg_1_0_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[1, 1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_1_tensor = int8_tensor
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 40
start = time.time()
results["time_low"] = paddle.concat(arg_1,axis=arg_2,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().astype(paddle.int64)
arg_1_1 = arg_1_1_tensor.clone().astype(paddle.int64)
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.concat(arg_1,axis=arg_2,)
results["time_high"] = time.time() - start

print(results)
