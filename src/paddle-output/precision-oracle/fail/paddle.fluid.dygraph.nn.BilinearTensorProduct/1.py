results = dict()
import paddle
import time
arg_1 = 5
arg_2 = 4
arg_3 = True
arg_class = paddle.fluid.dygraph.nn.BilinearTensorProduct(input1_dim=arg_1,input2_dim=arg_2,output_dim=arg_3,)
arg_4_0_tensor = paddle.rand([5, 5], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4_1_tensor = paddle.rand([5, 4], dtype=paddle.float32)
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
