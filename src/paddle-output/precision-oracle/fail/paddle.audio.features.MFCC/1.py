results = dict()
import paddle
import time
arg_1 = 44100
arg_2 = 40
arg_class = paddle.audio.features.MFCC(sr=arg_1,n_mfcc=arg_2,)
arg_3_0_tensor = paddle.randint(-16384,128,[1, 220500], dtype=paddle.float16)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().type(paddle.float32)
arg_3 = [arg_3_0,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
