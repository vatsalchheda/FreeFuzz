import paddle
arg_1_tensor = paddle.rand([-1, 4, 5, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([-1, 16, 5, 5], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([-1, 3], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.rand([-1, 5, 4, 4], dtype=paddle.float32)
arg_4 = arg_4_tensor.clone()
arg_5_tensor = paddle.rand([-1, 5, 10, 4], dtype=paddle.float32)
arg_5 = arg_5_tensor.clone()
arg_6 = 6000
arg_7 = 1000
arg_8 = -15.5
arg_9 = -21.9
arg_10 = 1.0
arg_11 = False
arg_12 = None
res = paddle.vision.ops.generate_proposals(scores=arg_1,bbox_deltas=arg_2,img_size=arg_3,anchors=arg_4,variances=arg_5,pre_nms_top_n=arg_6,post_nms_top_n=arg_7,nms_thresh=arg_8,min_size=arg_9,eta=arg_10,return_rois_num=arg_11,name=arg_12,)
