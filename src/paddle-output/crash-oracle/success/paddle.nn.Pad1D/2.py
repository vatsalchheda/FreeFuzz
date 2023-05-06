import paddle
arg_1 = -10
arg_2 = "replicate"
arg_class = paddle.nn.Pad1D(arg_1,mode=arg_2,)
arg_3_0_tensor = paddle.rand([1, 80, 137], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
