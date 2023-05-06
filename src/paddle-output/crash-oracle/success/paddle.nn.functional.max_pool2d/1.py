import paddle
arg_1_tensor = paddle.rand([64, 6, 28, 28], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = 2
arg_4 = 0
arg_5 = False
arg_6 = True
arg_7 = "NCHW"
arg_8 = None
res = paddle.nn.functional.max_pool2d(arg_1,kernel_size=arg_2,stride=arg_3,padding=arg_4,return_mask=arg_5,ceil_mode=arg_6,data_format=arg_7,name=arg_8,)
