import paddle
arg_1_tensor = paddle.rand([2, 4, 8, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([4, 6, 3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([6], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = 0
arg_5 = 0
arg_6_0 = 1
arg_6_1 = 1
arg_6 = [arg_6_0,arg_6_1,]
arg_7_0 = 1
arg_7_1 = 1
arg_7 = [arg_7_0,arg_7_1,]
arg_8 = 1
arg_9 = None
arg_10 = "NCHW"
res = paddle.nn.functional.conv2d_transpose(arg_1,arg_2,bias=arg_3,padding=arg_4,output_padding=arg_5,stride=arg_6,dilation=arg_7,groups=arg_8,output_size=arg_9,data_format=arg_10,)
