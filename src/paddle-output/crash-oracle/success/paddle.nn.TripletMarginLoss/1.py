import paddle
arg_1 = 1.0
arg_2 = True
arg_3 = "mean"
arg_class = paddle.nn.TripletMarginLoss(margin=arg_1,swap=arg_2,reduction=arg_3,)
arg_4_0_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_4_1 = arg_4_1_tensor.clone()
arg_4_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_4_2 = arg_4_2_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
res = arg_class(*arg_4)
