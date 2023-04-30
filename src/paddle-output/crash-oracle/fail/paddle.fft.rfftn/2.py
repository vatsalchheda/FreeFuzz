import paddle
arg_1_tensor = paddle.randint(-32,32768,[9, 7, 7, 3, 7], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = None
arg_4 = "backward"
res = paddle.fft.rfftn(arg_1,arg_2,arg_3,arg_4,)
