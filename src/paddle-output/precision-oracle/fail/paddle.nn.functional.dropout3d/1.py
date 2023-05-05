results = dict()
import paddle
import time
float_tensor = paddle.rand([2, 3, 4, 5, 6], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 8.0
start = time.time()
results["time_low"] = paddle.nn.functional.dropout3d(arg_1,training=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.nn.functional.dropout3d(arg_1,training=arg_2,)
results["time_high"] = time.time() - start

print(results)
