import paddle
arg_1_tensor = paddle.randint(-8,1024,[5, 48, 9, 9, 7], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = 3
arg_4 = "backward"
res = paddle.fft.rfft(arg_1,arg_2,arg_3,arg_4,)
