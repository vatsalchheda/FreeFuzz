import paddle
real = paddle.rand([1024, 0], paddle.float64)
imag = paddle.rand([1024, 0], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = True
res = paddle.linalg.lu(arg_1,get_infos=arg_2,)
