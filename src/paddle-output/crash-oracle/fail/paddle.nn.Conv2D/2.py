import paddle
arg_1 = 6
arg_2 = 16
arg_3 = -1024
arg_4 = 1
arg_5 = 0
arg_class = paddle.nn.Conv2D(arg_1,arg_2,arg_3,stride=arg_4,padding=arg_5,)
arg_6_0_tensor = paddle.rand([64, 6, 14, 14], dtype=paddle.float32)
arg_6_0 = arg_6_0_tensor.clone()
arg_6 = [arg_6_0,]
res = arg_class(*arg_6)
