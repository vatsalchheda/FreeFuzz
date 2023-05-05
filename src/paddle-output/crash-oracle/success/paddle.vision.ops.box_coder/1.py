import paddle
arg_1_tensor = paddle.rand([512, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 0.1
arg_2_1 = 0.1
arg_2_2 = 0.2
arg_2_3 = 0.2
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3_tensor = paddle.rand([512, 81, 4], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = "decode_center_size"
arg_5 = False
arg_6 = 1
arg_7 = None
res = paddle.vision.ops.box_coder(prior_box=arg_1,prior_box_var=arg_2,target_box=arg_3,code_type=arg_4,box_normalized=arg_5,axis=arg_6,name=arg_7,)
