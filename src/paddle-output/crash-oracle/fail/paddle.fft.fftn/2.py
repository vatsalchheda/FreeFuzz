import paddle
arg_1_tensor = paddle.randint(-4,1,[4, 3, 4, 2, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = 0
arg_3_1 = 1
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "ortho"
arg_5 = None
res = paddle.fft.fftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
