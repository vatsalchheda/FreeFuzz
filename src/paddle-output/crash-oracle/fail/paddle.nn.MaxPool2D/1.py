import paddle
arg_1 = 2
arg_2 = -31
arg_class = paddle.nn.MaxPool2D(arg_1,arg_2,)
arg_3_0_tensor = paddle.rand([64, 16, 10, 10], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
