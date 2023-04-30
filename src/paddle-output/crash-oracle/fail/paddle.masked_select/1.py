import paddle
arg_1_tensor = paddle.randint(-64,16384,[3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(0,2,[3, 4], dtype=paddle.bool)
arg_2 = arg_2_tensor.clone()
res = paddle.masked_select(arg_1,arg_2,)
