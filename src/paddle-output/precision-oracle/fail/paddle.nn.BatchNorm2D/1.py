results = dict()
import paddle
import time
arg_1 = 2090
arg_class = paddle.nn.BatchNorm2D(arg_1,)
int_tensor = paddle.randint(low=-128, high=127, shape=[1, 153, 271, 32], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_0_tensor = int8_tensor
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_2_0 = arg_2_0_tensor.clone().type(paddle.int64)
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)
