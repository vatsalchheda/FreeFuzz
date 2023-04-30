results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-128,1,[-1, 1], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,32768,[-1, 1], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-32,4,[-1, 4], dtype=paddle.int8)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.randint(-16,8192,[-1, 4], dtype=paddle.float16)
arg_4 = arg_4_tensor.clone()
arg_5 = 4
arg_6 = 1
start = time.time()
results["time_low"] = paddle.fluid.layers.beam_search(pre_ids=arg_1,pre_scores=arg_2,ids=arg_3,scores=arg_4,beam_size=arg_5,end_id=arg_6,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.int64)
arg_4 = arg_4_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.layers.beam_search(pre_ids=arg_1,pre_scores=arg_2,ids=arg_3,scores=arg_4,beam_size=arg_5,end_id=arg_6,)
results["time_high"] = time.time() - start

print(results)
