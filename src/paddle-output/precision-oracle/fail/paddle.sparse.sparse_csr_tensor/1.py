results = dict()
import paddle
import time
arg_1_0 = -62
arg_1_1 = 40
arg_1_2 = 1
arg_1_3 = -19
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2_0 = 1
arg_2_1 = 2
arg_2_2 = 0
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3_0 = 0.2
arg_3_1 = 0.4
arg_3_2 = 0.6
arg_3_3 = 0.8
arg_3_4 = 1.0
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,arg_3_4,]
arg_4_0 = -55
arg_4_1 = 26
arg_4 = [arg_4_0,arg_4_1,]
start = time.time()
results["time_low"] = paddle.sparse.sparse_csr_tensor(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,arg_3_4,]
arg_4 = [arg_4_0,arg_4_1,]
start = time.time()
results["time_high"] = paddle.sparse.sparse_csr_tensor(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)
