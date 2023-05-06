import paddle
arg_1 = 1024.0
arg_class = paddle.nn.CELU(arg_1,)
arg_2_0_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
