results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8192,8192,[-1, 100, 256], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-256,1,[1, 100, 150], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-32768,4,[1, 100, 150], dtype=paddle.float16)
arg_3 = arg_3_tensor.clone()
arg_4 = 12
arg_5 = 150
arg_6 = 1
arg_7 = 0.2
start = time.time()
results["time_low"] = paddle.fluid.layers.lstm(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,dropout_prob=arg_7,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.layers.lstm(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,dropout_prob=arg_7,)
results["time_high"] = time.time() - start

print(results)
