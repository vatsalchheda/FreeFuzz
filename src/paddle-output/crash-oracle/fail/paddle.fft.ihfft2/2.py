import paddle
arg_1_tensor = paddle.randint(-32768,1024,[6, 9, 5, 9, 7], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = 127.0
arg_3_1 = 1075.0
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "ortho"
res = paddle.fft.ihfft2(arg_1,arg_2,arg_3,arg_4,)
