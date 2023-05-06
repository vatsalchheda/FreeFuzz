import paddle
real = paddle.rand([36, 4], paddle.float32)
imag = paddle.rand([36, 4], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
res = paddle.nn.functional.channel_shuffle(arg_1,arg_2,)
