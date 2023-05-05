results = dict()
import paddle
import time
arg_class = paddle.nn.Layer()
arg_1_0_tensor = paddle.rand([5, 5], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[5, 5], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_1_tensor = int8_tensor
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_low"] = arg_class(*arg_1)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().astype(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().astype(paddle.int8)
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = arg_class(*arg_1)
results["time_high"] = time.time() - start

print(results)
