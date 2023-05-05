results = dict()
import paddle
import time
arg_class = paddle.nn.LogSigmoid()
float_tensor = paddle.rand([4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_0_tensor = f16_tensor
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
start = time.time()
results["time_low"] = arg_class(*arg_1)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1 = [arg_1_0,]
start = time.time()
results["time_high"] = arg_class(*arg_1)
results["time_high"] = time.time() - start

print(results)
