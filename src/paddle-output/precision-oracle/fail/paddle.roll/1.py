results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_tensor = int8_tensor
arg_2 = arg_2_tensor.clone()
arg_3_0 = True
arg_3 = [arg_3_0,]
arg_4 = None
start = time.time()
results["time_low"] = paddle.roll(arg_1,arg_2,arg_3,name=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float64)
arg_2 = arg_2_tensor.clone().astype(paddle.int32)
arg_3 = [arg_3_0,]
start = time.time()
results["time_high"] = paddle.roll(arg_1,arg_2,arg_3,name=arg_4,)
results["time_high"] = time.time() - start

print(results)
