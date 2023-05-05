import paddle
arg_1_tensor = paddle.randint(-2,8,[13], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16384,16384,[11], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-4,4,[4], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
arg_4_0 = 2
arg_4_1 = 2
arg_4 = [arg_4_0,arg_4_1,]
arg_5 = "zeros"
res = paddle.incubate.graph_khop_sampler(arg_1,arg_2,arg_3,arg_4,arg_5,)
