results = dict()
import paddle
import time
float_tensor = paddle.rand([0, 64], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([-1, 512], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
float_tensor = paddle.rand([-1, 512], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_tensor = f16_tensor
arg_3 = arg_3_tensor.clone()
start = time.time()
results["time_low"] = paddle.fluid.layers.lstm_unit(x_t=arg_1,hidden_t_prev=arg_2,cell_t_prev=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.layers.lstm_unit(x_t=arg_1,hidden_t_prev=arg_2,cell_t_prev=arg_3,)
results["time_high"] = time.time() - start

print(results)
