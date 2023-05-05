import paddle
arg_1 = -1
arg_2 = 10
arg_class = paddle.nn.Linear(arg_1,arg_2,)
arg_3_0_tensor = paddle.rand([64, 84], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
