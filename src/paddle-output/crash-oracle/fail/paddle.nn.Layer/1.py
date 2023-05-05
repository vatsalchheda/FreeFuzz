import paddle
arg_class = paddle.nn.Layer()
arg_1_0_tensor = paddle.rand([64, 16, 10, 10], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
res = arg_class(*arg_1)
