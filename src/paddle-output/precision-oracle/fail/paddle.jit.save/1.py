results = dict()
import paddle
import time
arg_1 = "matching"
arg_2 = "__internal_testing__/tiny-random-rocketqa-cross-encoder\static\inference"
start = time.time()
results["time_low"] = paddle.jit.save(arg_1,arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.jit.save(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
