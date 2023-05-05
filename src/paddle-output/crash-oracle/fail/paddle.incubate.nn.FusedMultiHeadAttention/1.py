import paddle
arg_1 = 8
arg_2 = 2
arg_class = paddle.incubate.nn.FusedMultiHeadAttention(arg_1,arg_2,)
arg_3_0_tensor = paddle.rand([2, 4, 128], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3_2_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_3_2 = arg_3_2_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
res = arg_class(*arg_3)
