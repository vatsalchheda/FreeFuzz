results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16384,8192,[-1, 4], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = 5
arg_4 = 4
arg_5 = 224
arg_6 = None
arg_7 = None
start = time.time()
results["time_low"] = paddle.vision.ops.distribute_fpn_proposals(fpn_rois=arg_1,min_level=arg_2,max_level=arg_3,refer_level=arg_4,refer_scale=arg_5,rois_num=arg_6,name=arg_7,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.vision.ops.distribute_fpn_proposals(fpn_rois=arg_1,min_level=arg_2,max_level=arg_3,refer_level=arg_4,refer_scale=arg_5,rois_num=arg_6,name=arg_7,)
results["time_high"] = time.time() - start

print(results)
