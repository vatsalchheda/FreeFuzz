import paddle
arg_1 = True
arg_2 = None
arg_3 = -5
arg_4 = 11.23606797749979
arg_5 = "leaky_relu"
arg_class = paddle.fluid.initializer.MSRAInitializer(uniform=arg_1,fan_in=arg_2,seed=arg_3,negative_slope=arg_4,nonlinearity=arg_5,)
arg_6_0_tensor = paddle.rand([512], dtype=paddle.float64)
arg_6_0 = arg_6_0_tensor.clone()
real = paddle.rand([2, 0], paddle.float64)
imag = paddle.rand([2, 0], paddle.float64)
arg_6_1_tensor = paddle.complex(real, imag)
arg_6_1 = arg_6_1_tensor.clone()
arg_6 = [arg_6_0,arg_6_1,]
res = arg_class(*arg_6)
