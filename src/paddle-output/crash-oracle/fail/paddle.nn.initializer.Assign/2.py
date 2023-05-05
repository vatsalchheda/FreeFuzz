import paddle
arg_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_class = paddle.nn.initializer.Assign(arg_1,)
real = paddle.rand([1], paddle.float32)
imag = paddle.rand([1], paddle.float32)
arg_2_0_tensor = paddle.complex(real, imag)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.rand([37], dtype=paddle.float64)
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
res = arg_class(*arg_2)
