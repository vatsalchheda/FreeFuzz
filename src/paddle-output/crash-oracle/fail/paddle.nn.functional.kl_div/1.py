import paddle
arg_1_tensor = paddle.rand([5, 20], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([0, 20, 1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
arg_3 = "mean"
res = paddle.nn.functional.kl_div(arg_1,arg_2,arg_3,)
