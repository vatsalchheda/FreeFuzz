import paddle
arg_1_tensor = paddle.randint(-32,8192,[4, 4, 4], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 38
arg_2_1 = 12
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.fft.fftn(arg_1,axes=arg_2,)
