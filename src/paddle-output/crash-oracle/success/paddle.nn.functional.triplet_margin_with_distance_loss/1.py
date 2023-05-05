import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = 1.0
arg_5 = False
arg_6 = "mean"
arg_7 = None
res = paddle.nn.functional.triplet_margin_with_distance_loss(arg_1,arg_2,arg_3,margin=arg_4,swap=arg_5,reduction=arg_6,name=arg_7,)
