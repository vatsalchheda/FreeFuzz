results = dict()
import paddle
import time
arg_1 = -42
int_tensor = paddle.randint(low=-128, high=128, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_tensor = int8_tensor
arg_2 = arg_2_tensor.clone()
arg_3 = 1
arg_4 = "paddleVarType"
start = time.time()
results["time_low"] = paddle.fluid.layers.tensor.range(arg_1,arg_2,arg_3,dtype=arg_4,)
results["time_low"] = time.time() - start
arg_2 = arg_2_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.fluid.layers.tensor.range(arg_1,arg_2,arg_3,dtype=arg_4,)
results["time_high"] = time.time() - start

print(results)
