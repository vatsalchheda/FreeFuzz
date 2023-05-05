results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = 4
arg_2_1 = 4
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = 45
arg_3_1 = -55
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
start = time.time()
results["time_low"] = paddle.fft.rfft2(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.fft.rfft2(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)
