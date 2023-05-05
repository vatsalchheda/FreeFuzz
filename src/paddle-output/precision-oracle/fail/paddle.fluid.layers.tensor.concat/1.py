results = dict()
import paddle
import time
int_tensor = paddle.randint(low=0, high=255, shape=[21, 23, 32, 1], dtype='int32')
uint8_tensor = int_tensor.astype('uint8')
arg_1_0_tensor = uint8_tensor
arg_1_0 = arg_1_0_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[2, 86, 32], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_1_tensor = int8_tensor
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = -26
start = time.time()
results["time_low"] = paddle.fluid.layers.tensor.concat(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().astype(paddle.uint8)
arg_1_1 = arg_1_1_tensor.clone().astype(paddle.int64)
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.fluid.layers.tensor.concat(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
