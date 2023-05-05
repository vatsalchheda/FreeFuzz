results = dict()
import paddle
import time
float_tensor = paddle.rand([4, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[4], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_tensor = int8_tensor
arg_2 = arg_2_tensor.clone()
arg_3 = 21
float_tensor = paddle.rand([4, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_tensor = f16_tensor
arg_4 = arg_4_tensor.clone()
float_tensor = paddle.rand([4, 1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_5_tensor = f16_tensor
arg_5 = arg_5_tensor.clone()
arg_6 = None
arg_7 = None
arg_8 = True
arg_9 = None
start = time.time()
results["time_low"] = paddle.nn.functional.hsigmoid_loss(arg_1,arg_2,arg_3,arg_4,arg_5,path_table=arg_6,path_code=arg_7,is_sparse=arg_8,name=arg_9,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.int64)
arg_4 = arg_4_tensor.clone().type(paddle.float32)
arg_5 = arg_5_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.hsigmoid_loss(arg_1,arg_2,arg_3,arg_4,arg_5,path_table=arg_6,path_code=arg_7,is_sparse=arg_8,name=arg_9,)
results["time_high"] = time.time() - start

print(results)
