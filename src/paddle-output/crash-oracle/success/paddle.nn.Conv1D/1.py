import paddle
arg_1 = 64
arg_2 = 1
arg_3 = 18
arg_4 = False
arg_class = paddle.nn.Conv1D(arg_1,arg_2,arg_3,bias_attr=arg_4,)
arg_5_0_tensor = paddle.rand([1, 64, 41100], dtype=paddle.float32)
arg_5_0 = arg_5_0_tensor.clone()
arg_5 = [arg_5_0,]
res = arg_class(*arg_5)
