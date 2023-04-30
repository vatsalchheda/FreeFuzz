import paddle
arg_1_tensor = paddle.randint(-2,64,[4, 4, 4], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = None
arg_4 = "forward"
res = paddle.fft.hfftn(arg_1,arg_2,arg_3,arg_4,)
