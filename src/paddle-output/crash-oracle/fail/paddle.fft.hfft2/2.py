import paddle
arg_1_tensor = paddle.randint(-256,128,[4, 4, 4], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = -2
arg_3_1 = -1
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "ortho"
res = paddle.fft.hfft2(arg_1,arg_2,arg_3,arg_4,)
