import paddle
arg_1 = 8
arg_class = paddle.nn.LayerNorm(arg_1,)
arg_2_0_tensor = paddle.rand([1, 1, 8], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
