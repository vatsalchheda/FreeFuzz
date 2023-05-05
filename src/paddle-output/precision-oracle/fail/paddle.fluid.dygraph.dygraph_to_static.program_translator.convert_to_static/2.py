results = dict()
import paddle
import time
arg_1 = "forward"
start = time.time()
results["time_low"] = paddle.fluid.dygraph.dygraph_to_static.program_translator.convert_to_static(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.fluid.dygraph.dygraph_to_static.program_translator.convert_to_static(arg_1,)
results["time_high"] = time.time() - start

print(results)
