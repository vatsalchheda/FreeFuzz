import paddle
float_tensor = paddle.rand([0, 61], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = -1
res = paddle.sort(arg_1,arg_2,)
