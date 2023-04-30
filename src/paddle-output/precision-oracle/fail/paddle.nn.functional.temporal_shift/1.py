results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1024,128,[6, 4, 2, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = -56.8
start = time.time()
results["time_low"] = paddle.nn.functional.temporal_shift(x=arg_1,seg_num=arg_2,shift_ratio=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.temporal_shift(x=arg_1,seg_num=arg_2,shift_ratio=arg_3,)
results["time_high"] = time.time() - start

print(results)
