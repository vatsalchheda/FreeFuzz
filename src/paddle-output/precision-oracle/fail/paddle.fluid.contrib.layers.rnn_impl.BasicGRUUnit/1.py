results = dict()
import paddle
import time
arg_1 = "basic_gru_reverse_layers_1"
arg_2 = 256
arg_3 = None
arg_4 = None
arg_5 = -39
arg_6 = None
arg_7 = "float32"
arg_class = paddle.fluid.contrib.layers.rnn_impl.BasicGRUUnit(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
arg_8_0_tensor = paddle.rand([20, 256], dtype=paddle.float32)
arg_8_0 = arg_8_0_tensor.clone()
arg_8_1_tensor = paddle.rand([-1, 256], dtype=paddle.float32)
arg_8_1 = arg_8_1_tensor.clone()
arg_8 = [arg_8_0,arg_8_1,]
start = time.time()
results["time_low"] = arg_class(*arg_8)
results["time_low"] = time.time() - start
arg_8_0 = arg_8_0_tensor.clone().astype(paddle.float32)
arg_8_1 = arg_8_1_tensor.clone().astype(paddle.float32)
arg_8 = [arg_8_0,arg_8_1,]
start = time.time()
results["time_high"] = arg_class(*arg_8)
results["time_high"] = time.time() - start

print(results)
