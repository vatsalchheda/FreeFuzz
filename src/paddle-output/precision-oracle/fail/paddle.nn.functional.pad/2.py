results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-128,4096,[1, 220500, 1], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 249
arg_2_1 = 193
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "reflect"
arg_4 = "NLC"
start = time.time()
results["time_low"] = paddle.nn.functional.pad(arg_1,pad=arg_2,mode=arg_3,data_format=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.nn.functional.pad(arg_1,pad=arg_2,mode=arg_3,data_format=arg_4,)
results["time_high"] = time.time() - start

print(results)
