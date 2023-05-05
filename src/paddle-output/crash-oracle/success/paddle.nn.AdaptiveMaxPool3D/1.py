import paddle
arg_1 = 28
arg_class = paddle.nn.AdaptiveMaxPool3D(output_size=arg_1,)
arg_2_0_tensor = paddle.rand([2, 3, 8, 32, 32], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
