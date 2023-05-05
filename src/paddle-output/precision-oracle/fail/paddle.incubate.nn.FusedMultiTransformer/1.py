results = dict()
import paddle
import time
arg_1 = 128
arg_2 = 2
arg_3 = 512
arg_4 = -1
arg_class = paddle.incubate.nn.FusedMultiTransformer(arg_1,arg_2,arg_3,num_layers=arg_4,)
float_tensor = paddle.rand([2, 4, 128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_5_0_tensor = f16_tensor
arg_5_0 = arg_5_0_tensor.clone()
float_tensor = paddle.rand([2, 1, 4, 4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_5_1_tensor = f16_tensor
arg_5_1 = arg_5_1_tensor.clone()
arg_5 = [arg_5_0,arg_5_1,]
start = time.time()
results["time_low"] = arg_class(*arg_5)
results["time_low"] = time.time() - start
arg_5_0 = arg_5_0_tensor.clone().type(paddle.float32)
arg_5_1 = arg_5_1_tensor.clone().type(paddle.float32)
arg_5 = [arg_5_0,arg_5_1,]
start = time.time()
results["time_high"] = arg_class(*arg_5)
results["time_high"] = time.time() - start

print(results)
