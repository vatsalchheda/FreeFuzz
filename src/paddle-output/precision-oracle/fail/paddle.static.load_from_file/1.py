results = dict()
import paddle
import time
arg_1 = "./infer_model.params"
start = time.time()
results["time_low"] = paddle.static.load_from_file(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.static.load_from_file(arg_1,)
results["time_high"] = time.time() - start

print(results)
