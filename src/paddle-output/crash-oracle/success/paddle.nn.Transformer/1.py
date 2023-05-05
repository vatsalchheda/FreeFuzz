import paddle
arg_1 = 128
arg_2 = 2
arg_3 = 67
arg_4 = 28
arg_5 = 512
arg_class = paddle.nn.Transformer(arg_1,arg_2,arg_3,arg_4,arg_5,)
arg_6_0_tensor = paddle.rand([2, 4, 128], dtype=paddle.float32)
arg_6_0 = arg_6_0_tensor.clone()
arg_6_1_tensor = paddle.rand([2, 6, 128], dtype=paddle.float32)
arg_6_1 = arg_6_1_tensor.clone()
arg_6_2_tensor = paddle.rand([2, 2, 4, 4], dtype=paddle.float32)
arg_6_2 = arg_6_2_tensor.clone()
arg_6_3_tensor = paddle.rand([2, 2, 6, 6], dtype=paddle.float32)
arg_6_3 = arg_6_3_tensor.clone()
arg_6_4_tensor = paddle.rand([2, 2, 6, 4], dtype=paddle.float32)
arg_6_4 = arg_6_4_tensor.clone()
arg_6 = [arg_6_0,arg_6_1,arg_6_2,arg_6_3,arg_6_4,]
res = arg_class(*arg_6)
