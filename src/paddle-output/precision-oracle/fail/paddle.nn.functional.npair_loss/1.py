results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8,512,[18, 6], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-256,2048,[18, 6], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-16384,32,[18], dtype=paddle.float16)
arg_3 = arg_3_tensor.clone()
arg_4 = 0.002
start = time.time()
results["time_low"] = paddle.nn.functional.npair_loss(arg_1,arg_2,arg_3,l2_reg=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.npair_loss(arg_1,arg_2,arg_3,l2_reg=arg_4,)
results["time_high"] = time.time() - start

print(results)
