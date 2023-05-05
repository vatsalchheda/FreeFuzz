results = dict()
import paddle
import time
real = paddle.rand([3], paddle.float32)
imag = paddle.rand([3], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4 = "mean"
arg_5 = None
arg_6 = None
start = time.time()
results["time_low"] = paddle.nn.functional.binary_cross_entropy_with_logits(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.complex128)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.binary_cross_entropy_with_logits(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,)
results["time_high"] = time.time() - start

print(results)
