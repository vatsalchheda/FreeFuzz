results = dict()
import paddle
import time
arg_1 = 2
arg_2 = 8
arg_class = paddle.nn.Embedding(arg_1,arg_2,)
int_tensor = paddle.randint(low=-128, high=127, shape=[1, 1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_3_0_tensor = int8_tensor
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().type(paddle.int64)
arg_3 = [arg_3_0,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
