import paddle
arg_1 = 2
arg_2 = 110.0
arg_class = paddle.nn.Pad1D(arg_1,mode=arg_2,)
arg_3_0_tensor = paddle.randint(-16384,2,[1, 51, 190, 1], dtype=paddle.bfloat16)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
