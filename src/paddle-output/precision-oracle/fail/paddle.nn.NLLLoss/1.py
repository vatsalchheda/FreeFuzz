results = dict()
import paddle
import time
arg_class = paddle.nn.NLLLoss()
float_tensor = paddle.rand([5, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_0_tensor = f16_tensor
arg_1_0 = arg_1_0_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[5], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_1_tensor = int8_tensor
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_low"] = arg_class(*arg_1)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().type(paddle.int64)
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = arg_class(*arg_1)
results["time_high"] = time.time() - start

print(results)
