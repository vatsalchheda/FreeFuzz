results = dict()
import paddle
import time
arg_1 = 256
arg_2 = 172
arg_class = paddle.fluid.dygraph.rnn.LSTMCell(arg_1,arg_2,)
arg_3_0_tensor = paddle.rand([64, 128], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([64, 256], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3_2_tensor = paddle.rand([64, 256], dtype=paddle.float32)
arg_3_2 = arg_3_2_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().astype(paddle.float64)
arg_3_1 = arg_3_1_tensor.clone().astype(paddle.float64)
arg_3_2 = arg_3_2_tensor.clone().astype(paddle.float64)
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
