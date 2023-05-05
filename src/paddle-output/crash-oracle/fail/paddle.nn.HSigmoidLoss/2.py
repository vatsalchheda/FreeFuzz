import paddle
arg_1 = True
arg_2 = 5
arg_class = paddle.nn.HSigmoidLoss(arg_1,arg_2,)
arg_3_0_tensor = paddle.rand([0, 1024, 1], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
real = paddle.rand([4], paddle.float64)
imag = paddle.rand([4], paddle.float64)
arg_3_1_tensor = paddle.complex(real, imag)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
res = arg_class(*arg_3)
