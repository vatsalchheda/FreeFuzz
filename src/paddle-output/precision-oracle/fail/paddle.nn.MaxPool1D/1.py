results = dict()
import paddle
import time
arg_1 = 2
arg_2 = 1
arg_3 = 12
arg_class = paddle.nn.MaxPool1D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
int_tensor = paddle.randint(low=-128, high=127, shape=[47, 3, 32], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_4_0_tensor = int8_tensor
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_4_0 = arg_4_0_tensor.clone().type(paddle.int16)
arg_4 = [arg_4_0,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)
