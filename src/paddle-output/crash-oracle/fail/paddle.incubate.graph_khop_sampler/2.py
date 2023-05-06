import paddle
arg_1_tensor = paddle.randint(-256, 8, [13], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1, 8192, [11], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-512, 512, [4], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
arg_4_0 = 16
arg_4_1 = 16
arg_4 = [arg_4_0,arg_4_1,]
arg_5 = False
res = paddle.incubate.graph_khop_sampler(arg_1,arg_2,arg_3,arg_4,arg_5,)
