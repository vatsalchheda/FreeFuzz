import paddle
arg_1_tensor = paddle.randint(-4,16384,[2, 12, 12], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 4
arg_2_1 = 5
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = 155.0
res = paddle.nn.functional.fold(arg_1,output_sizes=arg_2,kernel_sizes=arg_3,)
