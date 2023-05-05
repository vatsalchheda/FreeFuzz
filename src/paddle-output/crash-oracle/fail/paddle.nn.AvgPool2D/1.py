import paddle
arg_1 = 2
arg_2 = -72
arg_3 = 0
arg_class = paddle.nn.AvgPool2D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
arg_4_0_tensor = paddle.rand([1, 3, 32, 32], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
res = arg_class(*arg_4)
