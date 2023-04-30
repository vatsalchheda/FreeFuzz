results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8,32,[1, 1, 2, 2, 3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,1,[1, 1, 2, 2, 3], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3 = 2
arg_4 = None
arg_5 = 0
arg_6 = "NCDHW"
arg_7 = None
arg_8 = None
start = time.time()
results["time_low"] = paddle.nn.functional.max_unpool3d(arg_1,arg_2,kernel_size=arg_3,stride=arg_4,padding=arg_5,data_format=arg_6,output_size=arg_7,name=arg_8,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.int32)
start = time.time()
results["time_high"] = paddle.nn.functional.max_unpool3d(arg_1,arg_2,kernel_size=arg_3,stride=arg_4,padding=arg_5,data_format=arg_6,output_size=arg_7,name=arg_8,)
results["time_high"] = time.time() - start

print(results)
