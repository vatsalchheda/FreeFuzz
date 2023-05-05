import paddle
arg_1_tensor = paddle.rand([1, 1, 11, 11], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 0
arg_2_1 = 0
arg_2_2 = 0
arg_2_3 = 1
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3 = "replicate"
arg_4 = 96.0
arg_5 = "NCHW"
arg_6 = None
res = paddle.nn.functional.pad(arg_1,pad=arg_2,mode=arg_3,value=arg_4,data_format=arg_5,name=arg_6,)
