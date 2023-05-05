results = dict()
import paddle
import time
arg_1 = True
arg_2 = 32
arg_3 = 6
arg_4 = 3
arg_5 = 2
arg_6 = 9
arg_class = paddle.nn.Conv1DTranspose(arg_1,arg_2,arg_3,arg_4,padding=arg_5,output_padding=arg_6,)
float_tensor = paddle.rand([1, 256, 680], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_7_0_tensor = f16_tensor
arg_7_0 = arg_7_0_tensor.clone()
arg_7 = [arg_7_0,]
start = time.time()
results["time_low"] = arg_class(*arg_7)
results["time_low"] = time.time() - start
arg_7_0 = arg_7_0_tensor.clone().type(paddle.float32)
arg_7 = [arg_7_0,]
start = time.time()
results["time_high"] = arg_class(*arg_7)
results["time_high"] = time.time() - start

print(results)
