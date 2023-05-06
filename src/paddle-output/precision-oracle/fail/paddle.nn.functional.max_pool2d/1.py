results = dict()
import paddle
import time
float_tensor = paddle.rand([64, 6, 28, 28], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = 2
arg_4 = 0
arg_5 = False
arg_6 = True
arg_7 = "NCHW"
arg_8 = None
start = time.time()
results["time_low"] = paddle.nn.functional.max_pool2d(arg_1,kernel_size=arg_2,stride=arg_3,padding=arg_4,return_mask=arg_5,ceil_mode=arg_6,data_format=arg_7,name=arg_8,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.max_pool2d(arg_1,kernel_size=arg_2,stride=arg_3,padding=arg_4,return_mask=arg_5,ceil_mode=arg_6,data_format=arg_7,name=arg_8,)
results["time_high"] = time.time() - start

print(results)
