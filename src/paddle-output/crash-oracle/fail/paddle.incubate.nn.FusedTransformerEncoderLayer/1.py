import paddle
arg_1 = 128
arg_2 = 7
arg_3 = 512
arg_class = paddle.incubate.nn.FusedTransformerEncoderLayer(arg_1,arg_2,arg_3,)
arg_4_0_tensor = paddle.rand([2, 4, 128], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4_1_tensor = paddle.rand([2, 2, 4, 4], dtype=paddle.float32)
arg_4_1 = arg_4_1_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,]
res = arg_class(*arg_4)
