import paddle
arg_1_tensor = paddle.rand([5, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-256,256,[2, 3], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-64,16384,[2], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.randint(-1024,32768,[2], dtype=paddle.int64)
arg_4 = arg_4_tensor.clone()
arg_5 = 31
arg_6 = "mean"
arg_7 = True
res = paddle.nn.functional.ctc_loss(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,norm_by_times=arg_7,)
