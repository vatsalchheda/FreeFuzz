import paddle
arg_class = paddle.nn.MarginRankingLoss()
arg_1_0_tensor = paddle.randint(-32768,4,[2, 2], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-1,1,[2, 2], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1_2_tensor = paddle.randint(-4096,512,[2, 2], dtype=paddle.float32)
arg_1_2 = arg_1_2_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
res = arg_class(*arg_1)
