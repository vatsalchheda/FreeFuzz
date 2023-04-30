results = dict()
import paddle
import time
arg_1 = "cond"
arg_2 = "body"
arg_3_0_tensor = paddle.randint(-2,16,[1], dtype=paddle.int8)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(-16,16,[2], dtype=paddle.float16)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = paddle.static.nn.while_loop(cond=arg_1,body=arg_2,loop_vars=arg_3,)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().type(paddle.int64)
arg_3_1 = arg_3_1_tensor.clone().type(paddle.float32)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.static.nn.while_loop(cond=arg_1,body=arg_2,loop_vars=arg_3,)
results["time_high"] = time.time() - start

print(results)
