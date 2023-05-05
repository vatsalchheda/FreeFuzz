results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_tensor = int8_tensor
arg_2 = arg_2_tensor.clone()
arg_3 = False
arg_4_tensor = paddle.randint(0,2,[1])
arg_4 = arg_4_tensor.clone()
start = time.time()
results["time_low"] = paddle.fluid.layers.control_flow.less_than(x=arg_1,y=arg_2,force_cpu=arg_3,cond=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = arg_2_tensor.clone().type(paddle.int64)
arg_4 = arg_4_tensor.clone().type(paddle.bool)
start = time.time()
results["time_high"] = paddle.fluid.layers.control_flow.less_than(x=arg_1,y=arg_2,force_cpu=arg_3,cond=arg_4,)
results["time_high"] = time.time() - start

print(results)
