import paddle
arg_1 = True
arg_2 = 32
arg_3 = 6
arg_4 = 3
arg_5 = 2
arg_6 = 9
arg_class = paddle.nn.Conv1DTranspose(arg_1,arg_2,arg_3,arg_4,padding=arg_5,output_padding=arg_6,)
arg_7_0_tensor = paddle.rand([1, 256, 680], dtype=paddle.float32)
arg_7_0 = arg_7_0_tensor.clone()
arg_7 = [arg_7_0,]
res = arg_class(*arg_7)
