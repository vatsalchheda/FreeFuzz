results = dict()
import paddle
import time
arg_1 = "./test.wav"
arg_2_tensor = paddle.rand([1, 8000], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 16000
start = time.time()
results["time_low"] = paddle.audio.save(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_2 = arg_2_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.audio.save(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
