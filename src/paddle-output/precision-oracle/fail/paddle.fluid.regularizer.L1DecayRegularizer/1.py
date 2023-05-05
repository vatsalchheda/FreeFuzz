results = dict()
import paddle
import time
arg_1 = 59.1
arg_class = paddle.fluid.regularizer.L1DecayRegularizer(regularization_coeff=arg_1,)
arg_2_0_tensor = paddle.rand([10], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.rand([10], dtype=paddle.float32)
arg_2_1 = arg_2_1_tensor.clone()
arg_2_2_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_2_2 = arg_2_2_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_2_0 = arg_2_0_tensor.clone().astype(paddle.float32)
arg_2_1 = arg_2_1_tensor.clone().astype(paddle.float32)
arg_2_2 = arg_2_2_tensor.clone().astype(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)
