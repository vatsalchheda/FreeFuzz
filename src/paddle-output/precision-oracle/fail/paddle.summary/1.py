results = dict()
import paddle
import time
arg_1 = "__main__LeNetDictInput"
arg_2 = "builtinsdict"
start = time.time()
results["time_low"] = paddle.summary(arg_1,input=arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.summary(arg_1,input=arg_2,)
results["time_high"] = time.time() - start

print(results)
