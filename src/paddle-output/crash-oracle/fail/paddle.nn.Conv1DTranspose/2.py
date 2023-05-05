import paddle
arg_1 = 6144
arg_2 = 1
arg_3 = 46
arg_class = paddle.nn.Conv1DTranspose(arg_1,arg_2,arg_3,)
arg_4_0_tensor = paddle.rand([1, 2, 4], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
res = arg_class(*arg_4)
