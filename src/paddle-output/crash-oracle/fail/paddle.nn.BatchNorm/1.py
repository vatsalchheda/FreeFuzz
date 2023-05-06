import paddle
arg_1 = -49
arg_class = paddle.nn.BatchNorm(arg_1,)
arg_2_0_tensor = paddle.rand([3, 0, 3, 1], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
