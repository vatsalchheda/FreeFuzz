results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-128,16,[5, 5], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = "reflect"
arg_3 = True
arg_4 = None
start = time.time()
results["time_low"] = paddle.nansum(arg_1,axis=arg_2,keepdim=arg_3,name=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nansum(arg_1,axis=arg_2,keepdim=arg_3,name=arg_4,)
results["time_high"] = time.time() - start

print(results)
