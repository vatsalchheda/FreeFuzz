import paddle
arg_class = paddle.nn.Swish()
arg_1_0_tensor = paddle.rand([1, 512, 2048], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
res = arg_class(*arg_1)
