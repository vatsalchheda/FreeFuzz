results = dict()
import paddle
import time
arg_1 = "reader"
arg_2 = 1
start = time.time()
results["time_low"] = paddle.batch(arg_1,batch_size=arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.batch(arg_1,batch_size=arg_2,)
results["time_high"] = time.time() - start

print(results)
