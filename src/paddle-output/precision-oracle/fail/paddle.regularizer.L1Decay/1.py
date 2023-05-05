results = dict()
import paddle
import time
arg_1 = 0.0001
arg_class = paddle.regularizer.L1Decay(arg_1,)
arg_2_0_tensor = paddle.rand([10, 16], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
real = paddle.rand([57, 10], paddle.float32)
imag = paddle.rand([57, 10], paddle.float32)
arg_2_1_tensor = paddle.complex(real, imag)
arg_2_1 = arg_2_1_tensor.clone()
arg_2_2_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_2_2 = arg_2_2_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_2_0 = arg_2_0_tensor.clone().astype(paddle.float64)
arg_2_1 = arg_2_1_tensor.clone().astype(paddle.complex128)
arg_2_2 = arg_2_2_tensor.clone().astype(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)
