import paddle
float_tensor = paddle.rand([28, 8], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64, 2, [10], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.gather(arg_1,arg_2,)
