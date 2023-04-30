results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16,8192,[-1, 784], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 10
arg_3 = "softmax"
start = time.time()
results["time_low"] = paddle.static.nn.fc(arg_1,size=arg_2,activation=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.static.nn.fc(arg_1,size=arg_2,activation=arg_3,)
results["time_high"] = time.time() - start

print(results)
