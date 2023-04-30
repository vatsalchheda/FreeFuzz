results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16,2,[1, 3, 32, 32], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = -14
arg_3 = 2
arg_4 = 0
arg_5 = True
arg_6 = False
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
