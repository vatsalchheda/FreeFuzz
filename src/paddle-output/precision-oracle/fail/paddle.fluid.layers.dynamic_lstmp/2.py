results = dict()
import paddle
import time
float_tensor = paddle.rand([-1, 2048], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 2048
arg_3 = 256
arg_4 = False
arg_5 = True
arg_6 = "tanh"
arg_7 = "tanh"
start = time.time()
results["time_low"] = paddle.fluid.layers.dynamic_lstmp(input=arg_1,size=arg_2,proj_size=arg_3,use_peepholes=arg_4,is_reverse=arg_5,cell_activation=arg_6,proj_activation=arg_7,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.layers.dynamic_lstmp(input=arg_1,size=arg_2,proj_size=arg_3,use_peepholes=arg_4,is_reverse=arg_5,cell_activation=arg_6,proj_activation=arg_7,)
results["time_high"] = time.time() - start

print(results)
