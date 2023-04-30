results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16,4096,[10, 4], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16384,1,[10, 4], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-1024,512,[2, 21, 4], dtype=paddle.float16)
arg_3 = arg_3_tensor.clone()
arg_4 = "decode_center_size"
start = time.time()
results["time_low"] = paddle.fluid.layers.detection.box_coder(prior_box=arg_1,prior_box_var=arg_2,target_box=arg_3,code_type=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.fluid.layers.detection.box_coder(prior_box=arg_1,prior_box_var=arg_2,target_box=arg_3,code_type=arg_4,)
results["time_high"] = time.time() - start

print(results)
