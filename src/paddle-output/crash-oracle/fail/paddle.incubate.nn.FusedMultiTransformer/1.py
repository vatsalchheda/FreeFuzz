import paddle
arg_1 = 109
arg_2 = 2
arg_3 = 512
arg_4 = 1
arg_class = paddle.incubate.nn.FusedMultiTransformer(arg_1,arg_2,arg_3,num_layers=arg_4,)
arg_5_0_tensor = paddle.rand([2, 4, 128], dtype=paddle.float32)
arg_5_0 = arg_5_0_tensor.clone()
arg_5_1_tensor = paddle.rand([2, 1, 4, 4], dtype=paddle.float32)
arg_5_1 = arg_5_1_tensor.clone()
arg_5 = [arg_5_0,arg_5_1,]
res = arg_class(*arg_5)
