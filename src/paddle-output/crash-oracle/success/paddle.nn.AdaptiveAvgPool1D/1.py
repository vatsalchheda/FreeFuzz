import paddle
arg_1 = 16
arg_class = paddle.nn.AdaptiveAvgPool1D(output_size=arg_1,)
arg_2_0_tensor = paddle.rand([1, 3, 32], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
