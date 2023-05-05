import paddle
real = paddle.rand([-1, 0, 128], paddle.float64)
imag = paddle.rand([-1, 0, 128], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2 = [arg_2_0,]
res = paddle.fluid.layers.nn.squeeze(arg_1,axes=arg_2,)
