import paddle
arg_1_tensor = paddle.randint(-128,8192,[4, 4, 4], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = -1
arg_4 = "backward"
res = paddle.fft.irfft(arg_1,arg_2,arg_3,arg_4,)
