results = dict()
import paddle
import time
arg_1 = "collectionsOrderedDict"
arg_2 = "C:\Users\phalt\AppData\Local\Temp\tmpn5rk4ak3\model.pdparams"
start = time.time()
results["time_low"] = paddle.save(arg_1,arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.save(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
