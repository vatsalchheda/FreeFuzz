import paddle
arg_1 = 6
arg_2 = 6
arg_class = paddle.nn.GroupNorm(num_channels=arg_1,num_groups=arg_2,)
arg_3_0_tensor = paddle.rand([2, 6, 2, 2], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
