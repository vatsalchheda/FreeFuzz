results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8192,16384,[3, 224, 224, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128,1,[3, 224, 224, 1], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.nn.functional.dice_loss(input=arg_1,label=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.nn.functional.dice_loss(input=arg_1,label=arg_2,)
results["time_high"] = time.time() - start

print(results)
