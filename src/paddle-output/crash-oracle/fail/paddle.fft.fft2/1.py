import paddle
float_tensor = paddle.rand([2, 2, 1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
res = paddle.fft.fft2(arg_1,)
