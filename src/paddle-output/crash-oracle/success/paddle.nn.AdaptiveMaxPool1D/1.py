import paddle
arg_1 = 20
arg_2 = True
arg_class = paddle.nn.AdaptiveMaxPool1D(output_size=arg_1,return_mask=arg_2,)
arg_3_0_tensor = paddle.rand([1, 3, 32], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
