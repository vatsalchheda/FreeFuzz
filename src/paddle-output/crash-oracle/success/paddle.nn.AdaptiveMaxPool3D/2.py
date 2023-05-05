import paddle
arg_1 = 3
arg_2 = True
arg_class = paddle.nn.AdaptiveMaxPool3D(output_size=arg_1,return_mask=arg_2,)
arg_3_0_tensor = paddle.rand([2, 3, 8, 32, 32], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
