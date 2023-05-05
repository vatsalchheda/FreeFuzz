results = dict()
import paddle
import time
float_tensor = paddle.rand([1, 1, 182, 182], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = False
arg_2_1 = "max"
arg_2_2 = True
arg_2_3 = "reflect"
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3 = "replicate"
arg_4 = 0.0
arg_5 = "NCHW"
arg_6 = None
start = time.time()
results["time_low"] = paddle.nn.functional.pad(arg_1,pad=arg_2,mode=arg_3,value=arg_4,data_format=arg_5,name=arg_6,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
start = time.time()
results["time_high"] = paddle.nn.functional.pad(arg_1,pad=arg_2,mode=arg_3,value=arg_4,data_format=arg_5,name=arg_6,)
results["time_high"] = time.time() - start

print(results)
