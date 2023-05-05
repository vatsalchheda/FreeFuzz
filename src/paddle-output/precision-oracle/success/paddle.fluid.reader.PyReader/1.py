results = dict()
import paddle
import time
arg_1 = -1
arg_2 = True
arg_class = paddle.fluid.reader.PyReader(capacity=arg_1,return_list=arg_2,)
arg_3 = []
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3 = []
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
