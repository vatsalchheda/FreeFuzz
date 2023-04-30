import paddle
arg_1_tensor = paddle.randint(-8,2048,[-1, 2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16384,1,[2, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-512,2048,[3], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
res = paddle.nn.functional.linear(x=arg_1,weight=arg_2,bias=arg_3,)
