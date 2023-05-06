results = dict()
import paddle
import time
arg_1_0 = "FLAGS_eager_delete_tensor_gb"
arg_1_1 = "FLAGS_check_nan_inf"
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_low"] = paddle.get_flags(arg_1,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.get_flags(arg_1,)
results["time_high"] = time.time() - start

print(results)
