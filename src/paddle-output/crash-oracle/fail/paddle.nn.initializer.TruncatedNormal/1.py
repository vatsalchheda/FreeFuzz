import paddle
arg_1 = -1024.0
arg_2 = 0.02
arg_class = paddle.nn.initializer.TruncatedNormal(mean=arg_1,std=arg_2,)
arg_3_0_tensor = paddle.rand([768, 3072], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
res = arg_class(*arg_3)
