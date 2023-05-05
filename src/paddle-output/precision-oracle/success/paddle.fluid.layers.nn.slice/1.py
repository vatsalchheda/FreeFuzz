results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([-1, -1, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -17
arg_2 = [arg_2_0,]
int_tensor = paddle.randint(low=-128, high=127, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_3_0_tensor = int8_tensor
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
int_tensor = paddle.randint(low=-128, high=127, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_4_0_tensor = int8_tensor
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
start = time.time()
results["time_low"] = paddle.fluid.layers.nn.slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
arg_2 = [arg_2_0,]
arg_3_0 = arg_3_0_tensor.clone().astype(paddle.int32)
arg_3 = [arg_3_0,]
arg_4_0 = arg_4_0_tensor.clone().astype(paddle.int32)
arg_4 = [arg_4_0,]
start = time.time()
results["time_high"] = paddle.fluid.layers.nn.slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,)
results["time_high"] = time.time() - start

print(results)
