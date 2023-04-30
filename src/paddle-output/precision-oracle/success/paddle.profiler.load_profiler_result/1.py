results = dict()
import paddle
import time
arg_1 = "test_export_protobuf.pb"
start = time.time()
results["time_low"] = paddle.profiler.load_profiler_result(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.profiler.load_profiler_result(arg_1,)
results["time_high"] = time.time() - start

print(results)
