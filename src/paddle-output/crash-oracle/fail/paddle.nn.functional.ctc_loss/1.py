import paddle
arg_1_tensor = paddle.randint(-32768,8192,[5, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8192,4,[2, 3], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-1024,256,[2], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.randint(-8192,256,[2], dtype=paddle.int64)
arg_4 = arg_4_tensor.clone()
arg_5 = 0
arg_6 = "none"
res = paddle.nn.functional.ctc_loss(arg_1,arg_2,arg_3,arg_4,blank=arg_5,reduction=arg_6,)
