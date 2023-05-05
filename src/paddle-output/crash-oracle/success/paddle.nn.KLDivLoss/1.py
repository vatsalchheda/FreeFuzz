import paddle
arg_1 = "sum"
arg_class = paddle.nn.KLDivLoss(reduction=arg_1,)
arg_2_0_tensor = paddle.rand([5, 20], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.rand([5, 20], dtype=paddle.float32)
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
res = arg_class(*arg_2)
