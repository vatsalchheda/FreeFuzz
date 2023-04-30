import paddle
arg_1_tensor = paddle.randint(-4096,256,[-1, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = 5
arg_4 = 4
arg_5 = 224
arg_6 = None
arg_7 = None
res = paddle.vision.ops.distribute_fpn_proposals(fpn_rois=arg_1,min_level=arg_2,max_level=arg_3,refer_level=arg_4,refer_scale=arg_5,rois_num=arg_6,name=arg_7,)
