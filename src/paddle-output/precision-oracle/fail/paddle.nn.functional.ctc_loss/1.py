results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-512,4,[5, 2, 3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,16,[2, 3], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-16,1,[2], dtype=paddle.int8)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.randint(-64,128,[2], dtype=paddle.int8)
arg_4 = arg_4_tensor.clone()
arg_5 = 0
arg_6 = "none"
start = time.time()
results["time_low"] = paddle.nn.functional.ctc_loss(arg_1,arg_2,arg_3,arg_4,blank=arg_5,reduction=arg_6,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.int32)
arg_3 = arg_3_tensor.clone().type(paddle.int64)
arg_4 = arg_4_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.nn.functional.ctc_loss(arg_1,arg_2,arg_3,arg_4,blank=arg_5,reduction=arg_6,)
results["time_high"] = time.time() - start

print(results)
