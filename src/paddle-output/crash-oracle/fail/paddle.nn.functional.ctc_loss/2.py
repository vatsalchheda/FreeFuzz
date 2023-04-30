import paddle
arg_1_tensor = paddle.randint(-1,128,[5, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,32768,[18], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-2048,64,[2], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.randint(-8,32768,[2], dtype=paddle.int64)
arg_4 = arg_4_tensor.clone()
arg_5 = 0
arg_6 = "sum"
arg_7 = False
res = paddle.nn.functional.ctc_loss(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,norm_by_times=arg_7,)
