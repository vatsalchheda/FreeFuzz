import paddle
arg_1 = True
arg_2 = 2
arg_3 = 5
arg_class = paddle.nn.AvgPool3D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
arg_4_0_tensor = paddle.rand([1, 2, 3, 32, 32], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
res = arg_class(*arg_4)
