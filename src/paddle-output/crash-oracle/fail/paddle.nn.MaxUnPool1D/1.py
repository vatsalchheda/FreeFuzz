import paddle
arg_1 = -1
arg_2_0 = 1
arg_2_1 = 2
arg_2_2 = 1
arg_2_3 = 1
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_class = paddle.nn.MaxUnPool1D(kernel_size=arg_1,padding=arg_2,)
arg_3_0_tensor = paddle.randint(-4096,64,[1, 3, 8], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(-2,512,[1, 3, 8], dtype=paddle.int32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
res = arg_class(*arg_3)
