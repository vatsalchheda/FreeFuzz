import paddle
arg_1_tensor = paddle.randint(-2048,2048,[5, 5, 8, 5, 6], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
arg_3 = 19.0
arg_4 = "ortho"
res = paddle.fft.ifft(arg_1,arg_2,arg_3,arg_4,)
