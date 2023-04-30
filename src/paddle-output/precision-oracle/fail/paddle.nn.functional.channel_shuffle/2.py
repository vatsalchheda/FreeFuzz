results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-4,64,[1, 6, 1, 1], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
arg_3 = "NCHW"
arg_4 = None
start = time.time()
results["time_low"] = paddle.nn.functional.channel_shuffle(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.channel_shuffle(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)
