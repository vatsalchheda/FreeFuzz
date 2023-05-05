import paddle
arg_1 = 0
arg_2 = "none"
arg_class = paddle.nn.CTCLoss(blank=arg_1,reduction=arg_2,)
arg_3_0_tensor = paddle.rand([5, 2, 3], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(-64,1024,[2, 3], dtype=paddle.int32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3_2_tensor = paddle.randint(-4,2048,[2], dtype=paddle.int64)
arg_3_2 = arg_3_2_tensor.clone()
arg_3_3_tensor = paddle.randint(-64,2048,[2], dtype=paddle.int64)
arg_3_3 = arg_3_3_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,]
res = arg_class(*arg_3)
