import paddle
arg_1_tensor = paddle.rand([-1, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 65
arg_3 = "max"
arg_4 = 1024
arg_5 = 231
arg_6 = None
arg_7 = 1026
res = paddle.vision.ops.distribute_fpn_proposals(fpn_rois=arg_1,min_level=arg_2,max_level=arg_3,refer_level=arg_4,refer_scale=arg_5,rois_num=arg_6,name=arg_7,)
