import paddle
arg_1_tensor = paddle.randint(-64, 1024, [-1, 13], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_0 = 20
arg_2_1 = 32
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.static.nn.embedding(arg_1,size=arg_2,)
