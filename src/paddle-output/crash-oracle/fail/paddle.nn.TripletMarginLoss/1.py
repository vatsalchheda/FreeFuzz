import paddle
arg_1 = True
arg_class = paddle.nn.TripletMarginLoss(reduction=arg_1,)
arg_2_0_tensor = paddle.rand([37, 34], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.rand([3, 3, 0], dtype=paddle.float32)
arg_2_1 = arg_2_1_tensor.clone()
arg_2_2_tensor = paddle.rand([0, 0, 1], dtype=paddle.float32)
arg_2_2 = arg_2_2_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
res = arg_class(*arg_2)
