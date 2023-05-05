import paddle
arg_1 = 512
arg_2 = 512
arg_3 = -54
arg_4 = 1
arg_5 = 0
arg_6 = 1
arg_7 = 1
arg_8 = "zeros"
arg_9 = None
arg_10 = None
arg_11 = "NCL"
arg_class = paddle.nn.Conv1D(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,)
arg_12_0_tensor = paddle.rand([1, 512, 123], dtype=paddle.float32)
arg_12_0 = arg_12_0_tensor.clone()
arg_12 = [arg_12_0,]
res = arg_class(*arg_12)
