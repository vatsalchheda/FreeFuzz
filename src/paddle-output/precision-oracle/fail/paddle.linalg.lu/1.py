results = dict()
import paddle
import time
real = paddle.rand([1024, 0], paddle.float32)
imag = paddle.rand([1024, 0], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = True
start = time.time()
results["time_low"] = paddle.linalg.lu(arg_1,get_infos=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.complex128)
start = time.time()
results["time_high"] = paddle.linalg.lu(arg_1,get_infos=arg_2,)
results["time_high"] = time.time() - start

print(results)
