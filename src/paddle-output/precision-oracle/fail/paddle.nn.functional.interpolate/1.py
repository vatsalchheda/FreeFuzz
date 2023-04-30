results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-2048,16,[2, 3, 6, 10], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 12
arg_2_1 = 12
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = None
arg_4 = "nearest"
arg_5 = False
arg_6 = 0
arg_7 = "NCHW"
arg_8 = None
start = time.time()
results["time_low"] = paddle.nn.functional.interpolate(arg_1,size=arg_2,scale_factor=arg_3,mode=arg_4,align_corners=arg_5,align_mode=arg_6,data_format=arg_7,name=arg_8,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.nn.functional.interpolate(arg_1,size=arg_2,scale_factor=arg_3,mode=arg_4,align_corners=arg_5,align_mode=arg_6,data_format=arg_7,name=arg_8,)
results["time_high"] = time.time() - start

print(results)
