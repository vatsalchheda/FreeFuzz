import paddle
arg_1 = 5
arg_class = paddle.nn.LocalResponseNorm(size=arg_1,)
arg_2_0_tensor = paddle.rand([3, 3, 112, 112], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
