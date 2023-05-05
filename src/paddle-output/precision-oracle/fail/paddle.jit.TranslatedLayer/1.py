results = dict()
import paddle
import time
arg_1 = "builtinsdict"
arg_2 = "builtinsdict"
arg_class = paddle.jit.TranslatedLayer(arg_1,arg_2,)
arg_3_0_tensor = paddle.rand([16, 784], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().astype(paddle.float32)
arg_3 = [arg_3_0,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
