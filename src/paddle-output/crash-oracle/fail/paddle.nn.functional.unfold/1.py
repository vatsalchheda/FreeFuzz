import paddle
arg_1_tensor = paddle.randint(-32768,4,[2, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 3
arg_2_1 = 3
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = -63
arg_4 = 1
arg_5 = 53
res = paddle.nn.functional.unfold(arg_1,arg_2,arg_3,arg_4,arg_5,)
