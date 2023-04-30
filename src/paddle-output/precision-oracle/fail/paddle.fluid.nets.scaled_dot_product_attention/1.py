results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-4096,64,[3, 5, 9], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,2,[3, 6, 9], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-128,16384,[3, 6, 10], dtype=paddle.float16)
arg_3 = arg_3_tensor.clone()
start = time.time()
results["time_low"] = paddle.fluid.nets.scaled_dot_product_attention(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.nets.scaled_dot_product_attention(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
