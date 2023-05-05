import paddle
arg_1 = 3
arg_class = paddle.nn.BatchNorm(arg_1,)
arg_2_0_tensor = paddle.rand([3, 10, 3, 7], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
