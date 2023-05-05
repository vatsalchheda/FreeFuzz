results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[6], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 74
arg_3 = True
arg_4 = True
start = time.time()
results["time_low"] = paddle.unique(arg_1,return_index=arg_2,return_inverse=arg_3,return_counts=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.unique(arg_1,return_index=arg_2,return_inverse=arg_3,return_counts=arg_4,)
results["time_high"] = time.time() - start

print(results)
