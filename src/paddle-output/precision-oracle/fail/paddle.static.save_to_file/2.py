results = dict()
import paddle
import time
arg_1 = "pruned"
arg_2 = "builtinsbytes"
start = time.time()
results["time_low"] = paddle.static.save_to_file(arg_1,arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.static.save_to_file(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
