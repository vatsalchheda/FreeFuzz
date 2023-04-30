results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16,2048,[2, 3, 4], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = "float64"
start = time.time()
results["time_low"] = paddle.nn.functional.log_softmax(arg_1,dtype=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.log_softmax(arg_1,dtype=arg_2,)
results["time_high"] = time.time() - start

print(results)
