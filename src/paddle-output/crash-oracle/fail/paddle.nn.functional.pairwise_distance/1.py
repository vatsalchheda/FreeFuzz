import paddle
real = paddle.rand([2, 3], paddle.float32)
imag = paddle.rand([2, 3], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 2
arg_4 = 1e-06
arg_5 = False
arg_6 = None
res = paddle.nn.functional.pairwise_distance(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,)
