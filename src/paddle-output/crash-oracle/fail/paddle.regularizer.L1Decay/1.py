import paddle
arg_1 = 0.0001
arg_class = paddle.regularizer.L1Decay(arg_1,)
arg_2_0_tensor = paddle.rand([10, 16], dtype=paddle.float64)
arg_2_0 = arg_2_0_tensor.clone()
real = paddle.rand([57, 10], paddle.float64)
imag = paddle.rand([57, 10], paddle.float64)
arg_2_1_tensor = paddle.complex(real, imag)
arg_2_1 = arg_2_1_tensor.clone()
arg_2_2_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_2_2 = arg_2_2_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
res = arg_class(*arg_2)
