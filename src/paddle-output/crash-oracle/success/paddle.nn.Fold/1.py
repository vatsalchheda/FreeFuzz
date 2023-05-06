import paddle
arg_1_0 = 4
arg_1_1 = 5
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 2
arg_class = paddle.nn.Fold(output_sizes=arg_1,kernel_sizes=arg_2,)
arg_3_0_tensor = paddle.rand([2, 12, 12], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
