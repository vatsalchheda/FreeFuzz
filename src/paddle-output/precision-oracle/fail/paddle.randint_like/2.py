results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=128, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = -5
arg_3 = 5
start = time.time()
results["time_low"] = paddle.randint_like(arg_1,low=arg_2,high=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int32)
start = time.time()
results["time_high"] = paddle.randint_like(arg_1,low=arg_2,high=arg_3,)
results["time_high"] = time.time() - start

print(results)
