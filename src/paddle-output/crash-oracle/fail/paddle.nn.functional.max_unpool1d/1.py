import paddle
arg_1_tensor = paddle.rand([1, 3, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768,16384,[1, 1, 2, 2, 3], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3 = 2
arg_4 = None
arg_5 = 0
arg_6 = "NCL"
arg_7 = 63.0
arg_8 = None
res = paddle.nn.functional.max_unpool1d(arg_1,arg_2,kernel_size=arg_3,stride=arg_4,padding=arg_5,data_format=arg_6,output_size=arg_7,name=arg_8,)
