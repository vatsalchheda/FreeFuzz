results = dict()
import paddle
import time
float_tensor = paddle.rand([-1, 4, 5, 5], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([-1, 16, 5, 5], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
float_tensor = paddle.rand([-1, 3], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_tensor = f16_tensor
arg_3 = arg_3_tensor.clone()
float_tensor = paddle.rand([-1, 5, 4, 4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_4_tensor = f16_tensor
arg_4 = arg_4_tensor.clone()
float_tensor = paddle.rand([-1, 5, 10, 4], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_5_tensor = f16_tensor
arg_5 = arg_5_tensor.clone()
arg_6 = 6000
arg_7 = "max"
arg_8 = 0.5
arg_9 = 17.1
arg_10 = 1.0
arg_11 = False
arg_12 = None
start = time.time()
results["time_low"] = paddle.vision.ops.generate_proposals(scores=arg_1,bbox_deltas=arg_2,img_size=arg_3,anchors=arg_4,variances=arg_5,pre_nms_top_n=arg_6,post_nms_top_n=arg_7,nms_thresh=arg_8,min_size=arg_9,eta=arg_10,return_rois_num=arg_11,name=arg_12,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
arg_4 = arg_4_tensor.clone().type(paddle.float32)
arg_5 = arg_5_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.vision.ops.generate_proposals(scores=arg_1,bbox_deltas=arg_2,img_size=arg_3,anchors=arg_4,variances=arg_5,pre_nms_top_n=arg_6,post_nms_top_n=arg_7,nms_thresh=arg_8,min_size=arg_9,eta=arg_10,return_rois_num=arg_11,name=arg_12,)
results["time_high"] = time.time() - start

print(results)
