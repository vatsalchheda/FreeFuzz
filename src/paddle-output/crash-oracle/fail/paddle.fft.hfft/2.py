import paddle
arg_1_tensor = paddle.randint(-16,1024,[4, 4, 4], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = -23
arg_4 = "backward"
res = paddle.fft.hfft(arg_1,arg_2,arg_3,arg_4,)
