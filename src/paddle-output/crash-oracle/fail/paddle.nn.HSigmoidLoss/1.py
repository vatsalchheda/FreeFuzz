import paddle
arg_1 = 21
arg_2 = 2
arg_class = paddle.nn.HSigmoidLoss(arg_1,arg_2,)
arg_3_0_tensor = paddle.rand([4, 3], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(-1, 8, [4], dtype=paddle.int64arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
res = arg_class(*arg_3)
