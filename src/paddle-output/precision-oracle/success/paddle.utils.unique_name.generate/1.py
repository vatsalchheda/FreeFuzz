results = dict()
import paddle
import time
arg_1 = "fc_1.tmp_2.state"
start = time.time()
results["time_low"] = paddle.utils.unique_name.generate(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.utils.unique_name.generate(arg_1,)
results["time_high"] = time.time() - start

print(results)
