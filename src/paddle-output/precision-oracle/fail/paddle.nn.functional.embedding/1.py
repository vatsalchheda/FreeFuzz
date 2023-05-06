results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[1, 1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([512, 8], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4 = False
arg_5 = None
start = time.time()
results["time_low"] = paddle.nn.functional.embedding(arg_1,weight=arg_2,padding_idx=arg_3,sparse=arg_4,name=arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.embedding(arg_1,weight=arg_2,padding_idx=arg_3,sparse=arg_4,name=arg_5,)
results["time_high"] = time.time() - start

print(results)
