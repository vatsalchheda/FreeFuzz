results = dict()
import paddle
import time
float_tensor = paddle.rand([1, 3, 8], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
int_tensor = paddle.randint(low=-128, high=128, shape=[1, 3, 8], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_tensor = int8_tensor
arg_2 = arg_2_tensor.clone()
arg_3 = 2
arg_4 = None
arg_5 = 0
arg_6 = "NCL"
arg_7 = None
arg_8 = None
start = time.time()
results["time_low"] = paddle.nn.functional.max_unpool1d(arg_1,arg_2,kernel_size=arg_3,stride=arg_4,padding=arg_5,data_format=arg_6,output_size=arg_7,name=arg_8,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.int32)
start = time.time()
results["time_high"] = paddle.nn.functional.max_unpool1d(arg_1,arg_2,kernel_size=arg_3,stride=arg_4,padding=arg_5,data_format=arg_6,output_size=arg_7,name=arg_8,)
results["time_high"] = time.time() - start

print(results)
