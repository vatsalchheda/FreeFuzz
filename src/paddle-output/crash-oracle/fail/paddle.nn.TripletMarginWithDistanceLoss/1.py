import paddle
arg_1 = "none"
arg_class = paddle.nn.TripletMarginWithDistanceLoss(reduction=arg_1,)
arg_2_0_tensor = paddle.rand([44, 1], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2_1 = arg_2_1_tensor.clone()
arg_2_2_tensor = paddle.rand([0, 3], dtype=paddle.float32)
arg_2_2 = arg_2_2_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
res = arg_class(*arg_2)
