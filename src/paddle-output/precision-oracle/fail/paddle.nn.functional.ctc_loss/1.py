results = dict()
import paddle
import time
float_tensor = paddle.rand([5, 2, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[2, 3], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_tensor = int8_tensor
arg_2 = arg_2_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[2], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_3_tensor = int8_tensor
arg_3 = arg_3_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[2], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_4_tensor = int8_tensor
arg_4 = arg_4_tensor.clone()
arg_5 = -1024
arg_6 = "none"
arg_7 = False
start = time.time()
results["time_low"] = paddle.nn.functional.ctc_loss(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,norm_by_times=arg_7,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.int32)
arg_3 = arg_3_tensor.clone().type(paddle.int64)
arg_4 = arg_4_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.nn.functional.ctc_loss(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,norm_by_times=arg_7,)
results["time_high"] = time.time() - start

print(results)
