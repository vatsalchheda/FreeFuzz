results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[-1, 1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([-1, 1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[-1, 4], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_3_tensor = int8_tensor
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.rand([0, 4], dtype=paddle.float32)
arg_4 = arg_4_tensor.clone()
arg_5 = "max"
arg_6 = 1
start = time.time()
results["time_low"] = paddle.fluid.layers.beam_search(pre_ids=arg_1,pre_scores=arg_2,ids=arg_3,scores=arg_4,beam_size=arg_5,end_id=arg_6,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.int64)
arg_2 = arg_2_tensor.clone().astype(paddle.float32)
arg_3 = arg_3_tensor.clone().astype(paddle.int64)
arg_4 = arg_4_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.layers.beam_search(pre_ids=arg_1,pre_scores=arg_2,ids=arg_3,scores=arg_4,beam_size=arg_5,end_id=arg_6,)
results["time_high"] = time.time() - start

print(results)
