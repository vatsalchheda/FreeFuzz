import paddle
arg_1 = 1
arg_2 = 0.25
arg_class = paddle.nn.PReLU(arg_1,arg_2,)
arg_3_0_tensor = paddle.rand([1, 2, 3, 4], dtype=paddle.float64)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
