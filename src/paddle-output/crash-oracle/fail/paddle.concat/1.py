import paddle
arg_1_0_tensor = paddle.randint(-32768,2048,[1, 49], dtype=paddle.int64)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-8,1024,[1, 1], dtype=paddle.int64)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 40
res = paddle.concat(arg_1,axis=arg_2,)
