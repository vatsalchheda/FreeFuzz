results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
real = paddle.rand([1, 1, 2, 2, 3], paddle.float32)
imag = paddle.rand([1, 1, 2, 2, 3], paddle.float32)
arg_2_tensor = paddle.complex(real, imag)
arg_2 = arg_2_tensor.clone()
arg_3 = -55
arg_4 = None
arg_5 = 0
arg_6 = "NCDHW"
arg_7 = None
arg_8 = None
start = time.time()
results["time_low"] = paddle.nn.functional.max_unpool3d(arg_1,arg_2,kernel_size=arg_3,stride=arg_4,padding=arg_5,data_format=arg_6,output_size=arg_7,name=arg_8,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
arg_2 = arg_2_tensor.clone().astype(paddle.complex128)
start = time.time()
results["time_high"] = paddle.nn.functional.max_unpool3d(arg_1,arg_2,kernel_size=arg_3,stride=arg_4,padding=arg_5,data_format=arg_6,output_size=arg_7,name=arg_8,)
results["time_high"] = time.time() - start

print(results)
