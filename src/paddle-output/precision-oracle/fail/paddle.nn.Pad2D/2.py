results = dict()
import paddle
import time
arg_1_0 = 0
arg_1_1 = 0
arg_1_2 = 0
arg_1_3 = 1
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = "replicate"
arg_class = paddle.nn.Pad2D(arg_1,mode=arg_2,)
float_tensor = paddle.rand([1, 1, 47, 47], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_0_tensor = f16_tensor
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_3_0 = arg_3_0_tensor.clone().type(paddle.float32)
arg_3 = [arg_3_0,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
