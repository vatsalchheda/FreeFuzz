results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-32,32,[], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,16,[], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3 = 5
arg_4 = 0
start = time.time()
results["time_low"] = paddle.fluid.layers.beam_search_decode(arg_1,arg_2,beam_size=arg_3,end_id=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.layers.beam_search_decode(arg_1,arg_2,beam_size=arg_3,end_id=arg_4,)
results["time_high"] = time.time() - start

print(results)
