import paddle
arg_1 = 2
arg_2 = -25
arg_3 = -11
arg_4 = True
arg_class = paddle.nn.MaxPool1D(kernel_size=arg_1,stride=arg_2,padding=arg_3,return_mask=arg_4,)
arg_5_0_tensor = paddle.rand([17, 3], dtype=paddle.float32)
arg_5_0 = arg_5_0_tensor.clone()
arg_5 = [arg_5_0,]
res = arg_class(*arg_5)
