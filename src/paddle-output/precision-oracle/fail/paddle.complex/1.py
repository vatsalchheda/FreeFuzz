results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-32768,16,[4, 4, 4], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096,64,[4, 4, 4], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.complex(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
arg_2 = arg_2_tensor.clone().type(paddle.float64)
start = time.time()
results["time_high"] = paddle.complex(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
