import paddle
arg_1_tensor = paddle.randint(-8192,4096,[18, 6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2048,8192,[18, 6], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-2048,32768,[18], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = "max"
res = paddle.nn.functional.npair_loss(arg_1,arg_2,arg_3,l2_reg=arg_4,)
