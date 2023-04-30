import paddle
arg_1_tensor = paddle.randint(-8192,32,[100, 3, 224, 224], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -12
arg_2_1 = -39
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = 1
arg_4 = -59
arg_5 = 1
res = paddle.nn.functional.unfold(arg_1,arg_2,arg_3,arg_4,arg_5,)
