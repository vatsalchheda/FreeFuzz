import paddle
arg_1_tensor = paddle.randint(-16,64,[2], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = "int32"
res = paddle.ones(shape=arg_1,dtype=arg_2,)
