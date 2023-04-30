results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-256,1,[5, 2, 3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024,512,[18], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-64,1,[2], dtype=paddle.int8)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.randint(-2,2,[2], dtype=paddle.int8)
arg_4 = arg_4_tensor.clone()
arg_5 = 0
arg_6 = "sum"
arg_7 = False
start = time.time()
results["time_low"] = paddle.nn.functional.ctc_loss(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,norm_by_times=arg_7,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.int64)
arg_4 = arg_4_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.nn.functional.ctc_loss(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,norm_by_times=arg_7,)
results["time_high"] = time.time() - start

print(results)
