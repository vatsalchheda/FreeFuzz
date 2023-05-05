results = dict()
import paddle
import time
arg_1 = 128
arg_2 = 2
arg_3 = 512
arg_class = paddle.incubate.nn.FusedTransformerEncoderLayer(arg_1,arg_2,arg_3,)
arg_4_0_tensor = paddle.rand([2, 4, 128], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4_1_tensor = paddle.rand([2, 2, 4, 4], dtype=paddle.float32)
arg_4_1 = arg_4_1_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_4_0 = arg_4_0_tensor.clone().astype(paddle.float32)
arg_4_1 = arg_4_1_tensor.clone().astype(paddle.float32)
arg_4 = [arg_4_0,arg_4_1,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)
