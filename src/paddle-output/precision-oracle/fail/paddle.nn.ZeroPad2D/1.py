results = dict()
import paddle
import time
arg_1_0 = 1
arg_1_1 = 0
arg_1_2 = 1
arg_1_3 = 2
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_class = paddle.nn.ZeroPad2D(padding=arg_1,)
float_tensor = paddle.rand([1, 1, 2, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_0_tensor = f16_tensor
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2_0 = arg_2_0_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)
