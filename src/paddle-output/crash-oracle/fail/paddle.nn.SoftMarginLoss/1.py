import paddle
arg_1 = "none"
arg_class = paddle.nn.SoftMarginLoss(reduction=arg_1,)
arg_2_0_tensor = paddle.rand([5, 5], dtype=paddle.float64)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.randint(-64, 4, [5, 5], dtype=paddle.int64arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
res = arg_class(*arg_2)
