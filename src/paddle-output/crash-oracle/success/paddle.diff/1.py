import paddle
arg_1_tensor = paddle.randint(-256,2,[4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096,64,[2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.diff(arg_1,append=arg_2,)
