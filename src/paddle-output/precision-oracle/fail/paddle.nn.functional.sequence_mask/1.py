results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[4], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 12
arg_3 = "circular"
arg_4 = None
start = time.time()
results["time_low"] = paddle.nn.functional.sequence_mask(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.nn.functional.sequence_mask(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)
