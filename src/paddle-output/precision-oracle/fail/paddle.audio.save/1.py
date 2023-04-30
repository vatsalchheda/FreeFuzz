results = dict()
import paddle
import time
arg_1 = "circular"
arg_2_tensor = paddle.randint(-4,128,[1, 8000], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3 = 16000
start = time.time()
results["time_low"] = paddle.audio.save(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.audio.save(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
