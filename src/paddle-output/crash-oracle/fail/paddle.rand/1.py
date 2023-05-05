import paddle
arg_1_0_tensor = paddle.randint(-16,256,[1], dtype=paddle.int32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-32,1,[1], dtype=paddle.int32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
res = paddle.rand(shape=arg_1,)
