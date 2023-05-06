import paddle
arg_1_tensor = paddle.randint(-128, 8192, [8, 7, 7, 3, 9, 0], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = None
arg_4 = "backward"
res = paddle.fft.rfft2(arg_1,arg_2,arg_3,arg_4,)
