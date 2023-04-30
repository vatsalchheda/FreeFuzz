results = dict()
import paddle
import time
arg_1 = "func"
arg_2_tensor = paddle.randint(-16,256,[2], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.incubate.optimizer.functional.minimize_lbfgs(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.incubate.optimizer.functional.minimize_lbfgs(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
