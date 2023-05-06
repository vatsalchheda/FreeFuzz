results = dict()
import paddle
import time
arg_1 = -37.0
arg_2 = 1e-05
arg_3 = 0.9
arg_4 = -17
arg_5 = None
arg_6 = "NCL"
arg_7 = None
arg_class = paddle.nn.BatchNorm1D(arg_1,epsilon=arg_2,momentum=arg_3,weight_attr=arg_4,bias_attr=arg_5,data_format=arg_6,use_global_stats=arg_7,)
float_tensor = paddle.rand([1, 1024, 961], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_8_0_tensor = f16_tensor
arg_8_0 = arg_8_0_tensor.clone()
arg_8 = [arg_8_0,]
start = time.time()
results["time_low"] = arg_class(*arg_8)
results["time_low"] = time.time() - start
arg_8_0 = arg_8_0_tensor.clone().type(paddle.float32)
arg_8 = [arg_8_0,]
start = time.time()
results["time_high"] = arg_class(*arg_8)
results["time_high"] = time.time() - start

print(results)
