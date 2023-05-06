import paddle
arg_1 = 2048
arg_class = paddle.nn.BatchNorm2D(arg_1,)
arg_2_0_tensor = paddle.rand([1, 256, 125, 16], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
