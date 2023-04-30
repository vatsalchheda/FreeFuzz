results = dict()
import paddle
import time
arg_1 = "cond_zoom"
arg_2 = "body_zoom"
arg_3_0_tensor = paddle.randint(-8,128,[1], dtype=paddle.int8)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(0,2,[1], dtype=paddle.bool)
arg_3_1 = arg_3_1_tensor.clone()
arg_3_2_tensor = paddle.randint(-16384,1,[1], dtype=paddle.float16)
arg_3_2 = arg_3_2_tensor.clone()
arg_3_3_tensor = paddle.randint(-2048,16384,[1], dtype=paddle.float16)
arg_3_3 = arg_3_3_tensor.clone()
arg_3_4_tensor = paddle.randint(-16,32768,[1], dtype=paddle.float16)
arg_3_4 = arg_3_4_tensor.clone()
arg_3_5_tensor = paddle.randint(-8192,2,[2], dtype=paddle.float16)
arg_3_5 = arg_3_5_tensor.clone()
arg_3_6_tensor = paddle.randint(-1,2048,[1], dtype=paddle.float16)
arg_3_6 = arg_3_6_tensor.clone()
arg_3_7_tensor = paddle.randint(-16384,32,[1], dtype=paddle.float16)
arg_3_7 = arg_3_7_tensor.clone()
arg_3_8_tensor = paddle.randint(-4,2048,[1], dtype=paddle.float16)
arg_3_8 = arg_3_8_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,arg_3_4,arg_3_5,arg_3_6,arg_3_7,arg_3_8,]
start = time.time()
results["time_low"] = paddle.static.nn.while_loop(cond=arg_1,body=arg_2,loop_vars=arg_3,)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().type(paddle.int64)
arg_3_1 = arg_3_1_tensor.clone().type(paddle.bool)
arg_3_2 = arg_3_2_tensor.clone().type(paddle.float32)
arg_3_3 = arg_3_3_tensor.clone().type(paddle.float32)
arg_3_4 = arg_3_4_tensor.clone().type(paddle.float32)
arg_3_5 = arg_3_5_tensor.clone().type(paddle.float32)
arg_3_6 = arg_3_6_tensor.clone().type(paddle.float32)
arg_3_7 = arg_3_7_tensor.clone().type(paddle.float32)
arg_3_8 = arg_3_8_tensor.clone().type(paddle.float32)
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,arg_3_4,arg_3_5,arg_3_6,arg_3_7,arg_3_8,]
start = time.time()
results["time_high"] = paddle.static.nn.while_loop(cond=arg_1,body=arg_2,loop_vars=arg_3,)
results["time_high"] = time.time() - start

print(results)
