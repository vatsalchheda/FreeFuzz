import paddle
arg_1_tensor = paddle.randint(-256,4096,[-1, 13], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1024
arg_2_1 = 274
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.static.nn.embedding(arg_1,size=arg_2,)
