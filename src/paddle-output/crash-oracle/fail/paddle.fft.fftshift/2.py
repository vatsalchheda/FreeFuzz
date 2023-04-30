import paddle
arg_1_tensor = paddle.randint(-2048,32768,[10, 10], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
res = paddle.fft.fftshift(arg_1,arg_2,)
