results = dict()
import paddle
import time
arg_1 = "C:\Users\phalt\AppData\Local\Temp\tmps5x4q_rr\finetune_static\model"
start = time.time()
results["time_low"] = paddle.jit.load(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.jit.load(arg_1,)
results["time_high"] = time.time() - start

print(results)
