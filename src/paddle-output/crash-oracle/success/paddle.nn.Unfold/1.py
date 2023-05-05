import paddle
arg_1_0 = 3
arg_1_1 = 3
arg_1 = [arg_1_0,arg_1_1,]
arg_class = paddle.nn.Unfold(kernel_sizes=arg_1,)
arg_2_0_tensor = paddle.rand([100, 3, 224, 224], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
